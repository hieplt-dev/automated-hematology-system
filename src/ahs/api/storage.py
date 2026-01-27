import os
import json
import boto3
from botocore.exceptions import ClientError
from loguru import logger

class StorageClient:
    def __init__(self):
        self.endpoint_url = os.getenv("S3_ENDPOINT_URL")
        self.access_key = os.getenv("S3_ACCESS_KEY")
        self.secret_key = os.getenv("S3_SECRET_KEY")
        self.bucket_name = os.getenv("S3_BUCKET_NAME")

        if not all([self.endpoint_url, self.access_key, self.secret_key, self.bucket_name]):
            logger.warning("S3 configuration missing. Storage features will be disabled.")
            self.s3 = None
            return

        try:
            self.s3 = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
            )
            # Ensure bucket exists
            self.ensure_bucket()
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self.s3 = None

    def ensure_bucket(self):
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            try:
                self.s3.create_bucket(Bucket=self.bucket_name)
                logger.info(f"Created bucket {self.bucket_name}")
            except Exception as e:
                logger.error(f"Failed to create bucket {self.bucket_name}: {e}")

    def upload_image(self, file_bytes, filename, content_type="image/png"):
        if not self.s3:
            return None
        try:
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=f"images/{filename}",
                Body=file_bytes,
                ContentType=content_type,
            )
            return f"images/{filename}"
        except Exception as e:
            logger.error(f"Failed to upload image {filename}: {e}")
            return None

    def upload_json(self, data, filename):
        if not self.s3:
            return None
        try:
            json_bytes = json.dumps(data).encode("utf-8")
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=f"predictions/{filename}",
                Body=json_bytes,
                ContentType="application/json",
            )
            return f"predictions/{filename}"
        except Exception as e:
            logger.error(f"Failed to upload JSON {filename}: {e}")
            return None

    def list_history(self, limit=50):
        if not self.s3:
            return []
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix="predictions/",
            )
            contents = response.get("Contents", [])
            # Sort by LastModified descending
            contents.sort(key=lambda x: x["LastModified"], reverse=True)
            
            history = []
            for obj in contents[:limit]:
                try:
                    # Get the prediction JSON
                    res = self.s3.get_object(Bucket=self.bucket_name, Key=obj["Key"])
                    data = json.loads(res["Body"].read().decode("utf-8"))
                    data["timestamp"] = obj["LastModified"].isoformat()
                    # Add image URL (presigned or relative path if proxying)
                    # For now, we'll return the key, and the frontend might ask backend to retrieve it or we serve it.
                    # Let's simple return the data which should contain the image_key if we saved it effectively.
                    history.append(data)
                except Exception as e:
                    logger.warning(f"Failed to read history item {obj['Key']}: {e}")
                    continue
            return history

        except Exception as e:
            logger.error(f"Failed to list history: {e}")
            return []

    def get_file_content(self, key):
        if not self.s3:
            return None
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=key)
            return response["Body"].read()
        except Exception as e:
            logger.error(f"Failed to get file {key}: {e}")
            return None
