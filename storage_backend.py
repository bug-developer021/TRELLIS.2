import mimetypes
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol


class StorageBackend(Protocol):
    def upload_file(self, local_path: str) -> str:
        """Upload a local file and return a public URL."""


@dataclass
class LocalStorageBackend:
    root_dir: Path
    public_base_url: str

    def upload_file(self, local_path: str) -> str:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        local_path = Path(local_path)
        unique_name = f"{uuid.uuid4().hex}_{local_path.name}"
        target_path = self.root_dir / unique_name
        shutil.copy2(local_path, target_path)
        return f"{self.public_base_url.rstrip('/')}/{target_path.name}"


@dataclass
class S3StorageBackend:
    bucket: str
    region: Optional[str] = None
    endpoint_url: Optional[str] = None
    prefix: str = ""
    public_base_url: Optional[str] = None
    acl: Optional[str] = "public-read"

    def __post_init__(self) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("boto3 is required for S3 uploads") from exc
        self._client = boto3.client("s3", region_name=self.region, endpoint_url=self.endpoint_url)

    def _build_object_key(self, local_path: str) -> str:
        extension = Path(local_path).suffix
        filename = f"{uuid.uuid4().hex}{extension}"
        if self.prefix:
            return f"{self.prefix.strip('/')}/{filename}"
        return filename

    def upload_file(self, local_path: str) -> str:
        key = self._build_object_key(local_path)
        extra_args = {}
        if self.acl:
            extra_args["ACL"] = self.acl
        content_type, _ = mimetypes.guess_type(local_path)
        if content_type:
            extra_args["ContentType"] = content_type
        self._client.upload_file(local_path, self.bucket, key, ExtraArgs=extra_args or None)
        if self.public_base_url:
            return f"{self.public_base_url.rstrip('/')}/{key}"
        if self.endpoint_url:
            return f"{self.endpoint_url.rstrip('/')}/{self.bucket}/{key}"
        region_segment = f".{self.region}" if self.region else ""
        return f"https://{self.bucket}.s3{region_segment}.amazonaws.com/{key}"


def get_storage_backend() -> StorageBackend:
    backend = os.environ.get("STORAGE_BACKEND", "local").lower()
    if backend == "s3":
        bucket = os.environ.get("STORAGE_S3_BUCKET")
        if not bucket:
            raise RuntimeError("STORAGE_S3_BUCKET must be set for S3 storage backend")
        return S3StorageBackend(
            bucket=bucket,
            region=os.environ.get("STORAGE_S3_REGION"),
            endpoint_url=os.environ.get("STORAGE_S3_ENDPOINT_URL"),
            prefix=os.environ.get("STORAGE_S3_PREFIX", ""),
            public_base_url=os.environ.get("STORAGE_S3_PUBLIC_BASE_URL"),
            acl=os.environ.get("STORAGE_S3_ACL", "public-read"),
        )

    root_dir = Path(os.environ.get("STORAGE_LOCAL_DIR", "/tmp/trellis_uploads"))
    public_base_url = os.environ.get("STORAGE_PUBLIC_BASE_URL", f"file://{root_dir}")
    return LocalStorageBackend(root_dir=root_dir, public_base_url=public_base_url)
