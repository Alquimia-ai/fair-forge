"""LakeFS corpus connector implementation."""

import lakefs
from lakefs.client import Client as LakeFSClient
from loguru import logger

from .base import CorpusConnector, RegulatoryDocument


class LakeFSCorpusConnector(CorpusConnector):
    """
    Load regulatory corpus from LakeFS storage.

    Reads all .md files from a specified path prefix in a LakeFS repository.
    """

    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        repo_id: str,
        corpus_prefix: str = "corpus/",
        branch_name: str = "main",
    ):
        """
        Initialize LakeFS corpus connector.

        Args:
            host: LakeFS server URL.
            username: LakeFS username.
            password: LakeFS password.
            repo_id: LakeFS repository ID.
            corpus_prefix: Path prefix for corpus files in LakeFS.
            branch_name: Branch name to use.

        Raises:
            ValueError: If required credentials are incomplete.
        """
        if not all([host, username, password, repo_id]):
            msg = "LakeFS credentials incomplete. Required: host, username, password, repo_id"
            raise ValueError(msg)

        self.client = LakeFSClient(
            username=username,
            password=password,
            host=host,
        )
        self.repo = lakefs.Repository(repository_id=repo_id, client=self.client)
        self.ref = self.repo.ref(branch_name)
        self.corpus_prefix = corpus_prefix

    def load_documents(self) -> list[RegulatoryDocument]:
        """
        Load all markdown files from LakeFS.

        Returns:
            List of RegulatoryDocument objects.
        """
        documents: list[RegulatoryDocument] = []

        logger.info(f"Loading corpus from LakeFS prefix: {self.corpus_prefix}")

        all_objects = list(self.ref.objects(prefix=self.corpus_prefix))
        md_files = [obj for obj in all_objects if obj.path.endswith(".md")]

        if not md_files:
            logger.warning(f"No .md files found in LakeFS prefix '{self.corpus_prefix}'")
            return documents

        for obj in md_files:
            remote_path = obj.path
            file_name = remote_path.split("/")[-1]

            with self.ref.object(remote_path).reader(mode="rb") as reader:
                text = reader.read().decode("utf-8")

            documents.append(
                RegulatoryDocument(
                    text=text,
                    source=file_name,
                )
            )

        logger.info(f"Loaded {len(documents)} document(s) from LakeFS")
        return documents


__all__ = ["LakeFSCorpusConnector"]
