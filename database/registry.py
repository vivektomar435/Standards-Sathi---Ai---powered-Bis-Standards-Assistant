"""
Document Registry for BIS Standards.

The registry keeps track of every BIS standard ingested into the system.

It stores document-level information such as:
- standard number
- year
- title
- filename
- document hash
- ingestion date
- page count
- chunk count
- embedding model
- ingestion status
- version

The registry is intentionally separate from ChromaDB:
- Registry = document management / ingestion tracking
- ChromaDB = vector storage / retrieval
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sqlite3


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_REGISTRY_PATH = Path("data") / "document_registry.db"


# ============================================================
# DOCUMENT REGISTRY
# ============================================================


class DocumentRegistry:
    """
    SQLite-backed registry for BIS standards.

    One row represents one ingested document/version.
    """

    def __init__(self, db_path=DEFAULT_REGISTRY_PATH):
        self.db_path = Path(db_path)

        # Make sure the parent directory exists.
        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_database()

    # ========================================================
    # DATABASE CONNECTION
    # ========================================================

    def _connect(self):
        """
        Create a SQLite connection.
        """

        connection = sqlite3.connect(
            str(self.db_path)
        )

        connection.row_factory = sqlite3.Row

        return connection

    # ========================================================
    # DATABASE INITIALIZATION
    # ========================================================

    def _initialize_database(self):
        """
        Create the registry table if it does not exist.
        """

        with self._connect() as connection:

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    standard_number TEXT NOT NULL,
                    standard_year TEXT,

                    title TEXT,

                    filename TEXT NOT NULL,
                    file_path TEXT,

                    document_hash TEXT NOT NULL,

                    ingestion_date TEXT NOT NULL,

                    page_count INTEGER DEFAULT 0,
                    chunk_count INTEGER DEFAULT 0,

                    embedding_model TEXT,

                    status TEXT NOT NULL DEFAULT 'pending',

                    version TEXT,

                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,

                    UNIQUE(
                        standard_number,
                        standard_year,
                        document_hash
                    )
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_documents_standard
                ON documents(
                    standard_number,
                    standard_year
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_documents_hash
                ON documents(document_hash)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_documents_status
                ON documents(status)
                """
            )

            connection.commit()

    # ========================================================
    # FILE HASH
    # ========================================================

    @staticmethod
    def calculate_file_hash(
        file_path,
        chunk_size=1024 * 1024,
    ):
        """
        Calculate SHA-256 hash of a file.

        SHA-256 is used so that we can detect whether a PDF
        has already been ingested or has changed.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        sha256 = hashlib.sha256()

        with file_path.open("rb") as file:

            while True:

                data = file.read(chunk_size)

                if not data:
                    break

                sha256.update(data)

        return sha256.hexdigest()

    # ========================================================
    # TIMESTAMP
    # ========================================================

    @staticmethod
    def _now():
        """
        Return current UTC timestamp in ISO format.
        """

        return datetime.now(
            timezone.utc
        ).isoformat()

    # ========================================================
    # REGISTER DOCUMENT
    # ========================================================

    def register_document(
        self,
        standard_number,
        standard_year,
        filename,
        file_path=None,
        title=None,
        page_count=0,
        chunk_count=0,
        embedding_model=None,
        status="pending",
        version=None,
        document_hash=None,
    ):
        """
        Register a BIS standard.

        If document_hash is not supplied and file_path exists,
        the hash is calculated automatically.

        Returns:
            dict containing the registered document.
        """

        if not standard_number:
            raise ValueError(
                "standard_number is required."
            )

        if not filename:
            raise ValueError(
                "filename is required."
            )

        # Calculate hash automatically when possible.
        if document_hash is None:

            if file_path:

                document_hash = self.calculate_file_hash(
                    file_path
                )

            else:

                raise ValueError(
                    "Either document_hash or file_path "
                    "must be provided."
                )

        timestamp = self._now()

        with self._connect() as connection:

            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO documents (
                    standard_number,
                    standard_year,
                    title,
                    filename,
                    file_path,
                    document_hash,
                    ingestion_date,
                    page_count,
                    chunk_count,
                    embedding_model,
                    status,
                    version,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    standard_number,
                    standard_year,
                    title,
                    filename,
                    str(file_path) if file_path else None,
                    document_hash,
                    timestamp,
                    page_count,
                    chunk_count,
                    embedding_model,
                    status,
                    version,
                    timestamp,
                    timestamp,
                ),
            )

            connection.commit()

            # Retrieve the existing/new record.
            row = connection.execute(
                """
                SELECT *
                FROM documents
                WHERE standard_number = ?
                  AND standard_year = ?
                  AND document_hash = ?
                """,
                (
                    standard_number,
                    standard_year,
                    document_hash,
                ),
            ).fetchone()

        return dict(row)

    # ========================================================
    # CHECK WHETHER DOCUMENT EXISTS
    # ========================================================

    def document_exists(
        self,
        standard_number,
        standard_year=None,
        document_hash=None,
    ):
        """
        Check whether a document already exists.

        If document_hash is supplied, the check is performed
        against the exact PDF content.

        Otherwise the check uses standard number/year.
        """

        with self._connect() as connection:

            if document_hash:

                row = connection.execute(
                    """
                    SELECT id
                    FROM documents
                    WHERE standard_number = ?
                      AND standard_year = ?
                      AND document_hash = ?
                    LIMIT 1
                    """,
                    (
                        standard_number,
                        standard_year,
                        document_hash,
                    ),
                ).fetchone()

            else:

                row = connection.execute(
                    """
                    SELECT id
                    FROM documents
                    WHERE standard_number = ?
                      AND (
                          standard_year = ?
                          OR ? IS NULL
                      )
                    LIMIT 1
                    """,
                    (
                        standard_number,
                        standard_year,
                        standard_year,
                    ),
                ).fetchone()

        return row is not None

    # ========================================================
    # FIND DOCUMENT
    # ========================================================

    def get_document(
        self,
        standard_number,
        standard_year=None,
    ):
        """
        Retrieve a document by standard number/year.

        Returns:
            dict or None
        """

        with self._connect() as connection:

            row = connection.execute(
                """
                SELECT *
                FROM documents
                WHERE standard_number = ?
                  AND (
                      standard_year = ?
                      OR ? IS NULL
                  )
                ORDER BY id DESC
                LIMIT 1
                """,
                (
                    standard_number,
                    standard_year,
                    standard_year,
                ),
            ).fetchone()

        if row is None:
            return None

        return dict(row)

    # ========================================================
    # GET BY ID
    # ========================================================

    def get_by_id(self, document_id):
        """
        Retrieve a document by registry ID.
        """

        with self._connect() as connection:

            row = connection.execute(
                """
                SELECT *
                FROM documents
                WHERE id = ?
                """,
                (document_id,),
            ).fetchone()

        if row is None:
            return None

        return dict(row)

    # ========================================================
    # UPDATE DOCUMENT
    # ========================================================

    def update_document(
        self,
        document_id,
        **fields,
    ):
        """
        Update document metadata.

        Only known registry fields are allowed.
        """

        allowed_fields = {
            "standard_number",
            "standard_year",
            "title",
            "filename",
            "file_path",
            "document_hash",
            "ingestion_date",
            "page_count",
            "chunk_count",
            "embedding_model",
            "status",
            "version",
        }

        updates = {}

        for key, value in fields.items():

            if key in allowed_fields:
                updates[key] = value

        if not updates:
            return self.get_by_id(document_id)

        updates["updated_at"] = self._now()

        assignments = ", ".join(
            f"{key} = ?"
            for key in updates
        )

        values = list(updates.values())

        values.append(document_id)

        with self._connect() as connection:

            connection.execute(
                f"""
                UPDATE documents
                SET {assignments}
                WHERE id = ?
                """,
                values,
            )

            connection.commit()

        return self.get_by_id(document_id)

    # ========================================================
    # UPDATE STATUS
    # ========================================================

    def update_status(
        self,
        document_id,
        status,
    ):
        """
        Update ingestion status.

        Example statuses:
            pending
            processing
            completed
            failed
        """

        return self.update_document(
            document_id,
            status=status,
        )

    # ========================================================
    # UPDATE INGESTION RESULT
    # ========================================================

    def update_ingestion_result(
        self,
        document_id,
        page_count=None,
        chunk_count=None,
        embedding_model=None,
        status="completed",
    ):
        """
        Update information after successful ingestion.
        """

        fields = {
            "status": status,
        }

        if page_count is not None:
            fields["page_count"] = page_count

        if chunk_count is not None:
            fields["chunk_count"] = chunk_count

        if embedding_model is not None:
            fields["embedding_model"] = embedding_model

        return self.update_document(
            document_id,
            **fields,
        )

    # ========================================================
    # LIST DOCUMENTS
    # ========================================================

    def list_documents(
        self,
        status=None,
        standard_number=None,
    ):
        """
        Return registered documents.

        Optional filters:
        - status
        - standard_number
        """

        query = """
            SELECT *
            FROM documents
            WHERE 1 = 1
        """

        parameters = []

        if status is not None:

            query += """
                AND status = ?
            """

            parameters.append(status)

        if standard_number is not None:

            query += """
                AND standard_number = ?
            """

            parameters.append(
                standard_number
            )

        query += """
            ORDER BY standard_number,
                     standard_year,
                     id
        """

        with self._connect() as connection:

            rows = connection.execute(
                query,
                parameters,
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    # ========================================================
    # COUNT DOCUMENTS
    # ========================================================

    def count_documents(
        self,
        status=None,
    ):
        """
        Count registered documents.
        """

        with self._connect() as connection:

            if status is None:

                row = connection.execute(
                    """
                    SELECT COUNT(*)
                    AS count
                    FROM documents
                    """
                ).fetchone()

            else:

                row = connection.execute(
                    """
                    SELECT COUNT(*)
                    AS count
                    FROM documents
                    WHERE status = ?
                    """,
                    (status,),
                ).fetchone()

        return row["count"]

    # ========================================================
    # DELETE DOCUMENT
    # ========================================================

    def delete_document(
        self,
        document_id,
    ):
        """
        Delete a registry entry.

        IMPORTANT:
        This only removes the registry record.
        It does NOT delete vectors from ChromaDB.
        """

        with self._connect() as connection:

            cursor = connection.execute(
                """
                DELETE FROM documents
                WHERE id = ?
                """,
                (document_id,),
            )

            connection.commit()

        return cursor.rowcount > 0

    # ========================================================
    # EXPORT REGISTRY
    # ========================================================

    def export_json(
        self,
        output_path,
    ):
        """
        Export the complete registry to JSON.

        Useful for debugging, reporting, and future migration.
        """

        documents = self.list_documents()

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                documents,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return output_path

    # ========================================================
    # DATABASE INFO
    # ========================================================

    def info(self):
        """
        Return basic registry statistics.
        """

        return {
            "database_path": str(
                self.db_path
            ),
            "document_count": self.count_documents(),
            "completed_count": self.count_documents(
                status="completed"
            ),
            "pending_count": self.count_documents(
                status="pending"
            ),
            "processing_count": self.count_documents(
                status="processing"
            ),
            "failed_count": self.count_documents(
                status="failed"
            ),
        }


# ============================================================
# BACKWARD-COMPATIBLE ALIAS
# ============================================================

Registry = DocumentRegistry