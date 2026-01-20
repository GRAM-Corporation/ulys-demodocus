# S3 Infrastructure Setup

Set up AWS S3 bucket and Google Drive sync for the GRAM deployment pipeline.

## Architecture

```
Google Drive (source videos) → rclone sync → S3 bucket → Presigned URLs → ElevenLabs
```

## Tasks

### 1. Create S3 Bucket

```bash
aws s3 mb s3://gram-deployments --region us-east-1
```

Bucket structure mirrors local `deployments/`:
```
gram-deployments/
├── deploy_20250119_vinci_01/
│   └── sources/
│       └── gopro_01/
│           └── GX010006.MP4
└── deploy_20250120_vinci_02/
    └── ...
```

### 2. Configure Bucket Policy

- Block public access (default)
- Enable versioning (optional, for safety)
- Set lifecycle rules to transition old deployments to Glacier (optional)

```bash
aws s3api put-bucket-versioning --bucket gram-deployments --versioning-configuration Status=Enabled
```

### 3. Create IAM User for GRAM

```bash
aws iam create-user --user-name gram-pipeline

aws iam put-user-policy --user-name gram-pipeline --policy-name gram-s3-access --policy-document '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::gram-deployments",
        "arn:aws:s3:::gram-deployments/*"
      ]
    }
  ]
}'

aws iam create-access-key --user-name gram-pipeline
```

Save the AccessKeyId and SecretAccessKey.

### 4. Install and Configure rclone

```bash
# Install rclone
curl https://rclone.org/install.sh | sudo bash

# Configure Google Drive remote
rclone config
# - Name: gdrive
# - Type: drive
# - Follow OAuth flow

# Configure S3 remote
rclone config
# - Name: s3
# - Type: s3
# - Provider: AWS
# - Enter access key and secret
# - Region: us-east-1
```

### 5. Test Sync

```bash
# List Google Drive deployments folder
rclone ls gdrive:Deployments/

# Sync a single deployment
rclone sync gdrive:Deployments/deploy_20250119_vinci_01 s3:gram-deployments/deploy_20250119_vinci_01 --progress

# Sync all deployments
rclone sync gdrive:Deployments s3:gram-deployments --progress
```

### 6. Verify Presigned URLs Work

```python
import boto3

s3 = boto3.client('s3',
    aws_access_key_id='YOUR_KEY',
    aws_secret_access_key='YOUR_SECRET',
    region_name='us-east-1'
)

url = s3.generate_presigned_url('get_object',
    Params={'Bucket': 'gram-deployments', 'Key': 'deploy_20250119_vinci_01/sources/gopro_01/GX010006.MP4'},
    ExpiresIn=3600
)

print(url)
# Test: curl -I "<url>" should return 200
```

### 7. Update .env

Add to project `.env`:
```
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
GRAM_S3_BUCKET=gram-deployments
```

### 8. Optional: Automated Sync

For continuous sync, set up a cron job or systemd timer:

```bash
# crontab -e
# Sync every hour
0 * * * * rclone sync gdrive:Deployments s3:gram-deployments --log-file=/var/log/rclone-gram.log
```

Or use AWS DataSync for managed solution.

## Verification Checklist

- [ ] S3 bucket created and accessible
- [ ] IAM user created with correct permissions
- [ ] rclone configured for both Google Drive and S3
- [ ] Test sync works for sample deployment
- [ ] Presigned URL generation works
- [ ] URL is fetchable (test with curl)
- [ ] .env updated with credentials

## Files

- `.env` - Add AWS credentials
- `scripts/sync-deployments.sh` - Optional sync script
