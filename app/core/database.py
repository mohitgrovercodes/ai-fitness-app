import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from app.utils.logger import logger

class ChromaDBManager:
    """
    Manages a shared ChromaDB client and ensures all operations happen
    in a SINGLE background thread to prevent Segfaults on Mac/Linux.
    
    CRITICAL: chromadb is imported ONLY inside the run_query method
    to prevent binary extension conflicts during graph orchestration.
    """
    _instance = None
    _client = None
    _executor = None
    _chroma_dir = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChromaDBManager, cls).__new__(cls)
            root_dir = Path(__file__).resolve().parent.parent.parent
            cls._chroma_dir = root_dir / "chromadb_store"
            cls._executor = ThreadPoolExecutor(max_workers=1)
            logger.info("✅ [ChromaDB Manager] Single-threaded executor initialized (Lazy DB Loading).")
        return cls._instance

    def _get_client(self):
        """Lazy initialization of the actual ChromaDB client."""
        if self._client is None:
            import chromadb # DEFERRED IMPORT
            try:
                self._client = chromadb.PersistentClient(path=str(self._chroma_dir))
                logger.info("✅ [ChromaDB Manager] PersistentClient initialized in background thread.")
            except Exception as e:
                logger.error(f"❌ [ChromaDB Manager] Client Init Error: {e}")
        return self._client

    async def run_query(self, collection_name: str, **kwargs):
        """Runs a query in the dedicated database thread with deferred imports."""
        loop = asyncio.get_running_loop()
        
        def _sync_query():
            client = self._get_client()
            if not client:
                return None
            collection = client.get_collection(collection_name)
            return collection.query(**kwargs)
        
        # Use the single-threaded executor for the actual C++ call
        return await loop.run_in_executor(self._executor, _sync_query)

# Singleton instance
db_manager = ChromaDBManager()