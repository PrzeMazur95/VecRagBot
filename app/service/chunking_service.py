"""
This service provides functionality to chunk text to smaller portions
"""
from typing import List, Optional
import tiktoken

from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkingService:
    """
    # if tikotken returns, that specific content have more than chunk_size tokens, it will divide it to smaller chunks.
    # By default, first it will try to chunk it by '\n\n' if it is still to big then by \n, ad still if to big by ' ',
    # and at the end it will chunk it by any character
    """

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding('cl100k_base')  # Same tokenizer, as OPENAI LLM uses to calc tokens

    def split_to_smaller_chunks(
        self,
        data: str,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None) \
            -> List[str]:
        """
        If tikotken returns, that specific content have more than chunk_size tokens, it will divide it to smaller chunks
        By default, first it will try to chunk it by '\n\n' if it is still to big then by \n, ad still if to big by ' ',
        and at the end it will chunk it by any character

        :param chunk_size:
        :param chunk_overlap:
        :param data: Page content
        :param separators: Default are : '\n\n', '\n', ' ', ''
        :return: List of smaller chunks
        """
        if separators is None:
            separators = ['\n\n', '\n', ' ', '']

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self.tokens_length,
            separators=separators
        )

        return text_splitter.split_text(data)

    def tokens_length(self, text: str) -> int:
        """
        Method to calculate given text and return how many tokens it takes
        :param text: str
        :return: int
        """
        tokens = self.tokenizer.encode(
            text,
            disallowed_special=()
        )

        return len(tokens)