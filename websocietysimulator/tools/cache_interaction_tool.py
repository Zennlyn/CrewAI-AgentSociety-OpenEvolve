import logging
import threading
import os
import json
import lmdb
from typing import Optional, Dict, List, Iterator
from tqdm import tqdm

logger = logging.getLogger("websocietysimulator")

class CacheInteractionTool:
    # Class-level registry for LMDB environments to prevent multiple opens in the same process
    _envs: Dict[str, lmdb.Environment] = {}
    _envs_lock = threading.Lock()

    def __init__(self, data_dir: str):
        """
        Initialize the tool with the dataset directory.
        Args:
            data_dir: Path to the directory containing Yelp dataset files.
        """
        logger.info(f"Initializing InteractionTool with data directory: {data_dir}")
        self.data_dir = data_dir

        # Create LMDB environments
        self.env_dir = os.path.join(data_dir, "lmdb_cache")
        os.makedirs(self.env_dir, exist_ok=True)

        with self._envs_lock:
            self.user_env = self._get_or_create_env("users", 2 * 1024 * 1024 * 1024)
            self.item_env = self._get_or_create_env("items", 2 * 1024 * 1024 * 1024)
            self.review_env = self._get_or_create_env("reviews", 8 * 1024 * 1024 * 1024)

        # Initialize the database if empty
        self._initialize_db()

    def _get_or_create_env(self, name: str, map_size: int) -> lmdb.Environment:
        """Get an existing LMDB environment or create a new one."""
        env_path = os.path.join(self.env_dir, name)
        abs_path = os.path.abspath(env_path)
        
        if abs_path not in self._envs:
            logger.info(f"Opening LMDB environment: {abs_path}")
            self._envs[abs_path] = lmdb.open(abs_path, map_size=map_size)
        return self._envs[abs_path]

    def _initialize_db(self):
        """Initialize the LMDB databases with data if they are empty."""
        # Initialize users
        with self.user_env.begin(write=True) as txn:
            if not txn.stat()['entries']:
                with txn.cursor() as cursor:
                    for user in tqdm(self._iter_file('user.json')):
                        cursor.put(
                            user['user_id'].encode(),
                            json.dumps(user).encode()
                        )

        # Initialize items
        with self.item_env.begin(write=True) as txn:
            if not txn.stat()['entries']:
                with txn.cursor() as cursor:
                    for item in tqdm(self._iter_file('item.json')):
                        cursor.put(
                            item['item_id'].encode(),
                            json.dumps(item).encode()
                        )

        # Initialize reviews and their indices
        with self.review_env.begin(write=True) as txn:
            if not txn.stat()['entries']:
                for review in tqdm(self._iter_file('review.json')):
                    # Store the review
                    txn.put(
                        review['review_id'].encode(),
                        json.dumps(review).encode()
                    )

                    # Update item reviews index (store only review_ids)
                    item_review_ids = json.loads(txn.get(f"item_{review['item_id']}".encode()) or '[]')
                    item_review_ids.append(review['review_id'])
                    txn.put(
                        f"item_{review['item_id']}".encode(),
                        json.dumps(item_review_ids).encode()
                    )

                    # Update user reviews index (store only review_ids)
                    user_review_ids = json.loads(txn.get(f"user_{review['user_id']}".encode()) or '[]')
                    user_review_ids.append(review['review_id'])
                    txn.put(
                        f"user_{review['user_id']}".encode(),
                        json.dumps(user_review_ids).encode()
                    )

    def _iter_file(self, filename: str) -> Iterator[Dict]:
        """Iterate through file line by line."""
        file_path = os.path.join(self.data_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                yield json.loads(line)

    def get_user(self, user_id: str) -> Optional[Dict]:
        """Fetch user data based on user_id."""
        with self.user_env.begin() as txn:
            user_data = txn.get(user_id.encode())
            if user_data:
                return json.loads(user_data)
        return None

    def get_item(self, item_id: str) -> Optional[Dict]:
        """Fetch item data based on item_id."""
        if not item_id:
            return None

        with self.item_env.begin() as txn:
            item_data = txn.get(item_id.encode())
            if item_data:
                return json.loads(item_data)
        return None

    def get_reviews(
            self,
            item_id: Optional[str] = None,
            user_id: Optional[str] = None,
            review_id: Optional[str] = None
    ) -> List[Dict]:
        """Fetch reviews filtered by various parameters."""
        if review_id:
            with self.review_env.begin() as txn:
                review_data = txn.get(review_id.encode())
                if review_data:
                    return [json.loads(review_data)]
            return []

        with self.review_env.begin() as txn:
            if item_id:
                review_ids = json.loads(txn.get(f"item_{item_id}".encode()) or '[]')
            elif user_id:
                review_ids = json.loads(txn.get(f"user_{user_id}".encode()) or '[]')
            else:
                return []

            # Fetch complete review data for each review_id
            reviews = []
            for rid in review_ids:
                review_data = txn.get(rid.encode())
                if review_data:
                    reviews.append(json.loads(review_data))
            return reviews

    # Removed __del__ to prevent accidental closing of shared environments