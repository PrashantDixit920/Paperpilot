from pathlib import Path
from tempfile import NamedTemporaryFile

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def load_document(uploaded_file):
    """
    Load a PDF, TXT, Markdown, or DOCX file.

    uploaded_file must provide:
        - name: the uploaded filename
        - getvalue(): the uploaded file's bytes

    Returns:
        A list of LangChain Document objects.
    """

    filename = Path(uploaded_file.name).name
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {extension}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # Save the uploaded bytes temporarily because the loaders
    # expect a filesystem path.
    with NamedTemporaryFile(
        suffix=extension,
        delete=False
    ) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(uploaded_file.getvalue())

    try:
        if extension == ".pdf":
            loader = PyPDFLoader(str(temp_path))

        elif extension in {".txt", ".md"}:
            loader = TextLoader(
                str(temp_path),
                encoding="utf-8"
            )

        elif extension == ".docx":
            loader = Docx2txtLoader(str(temp_path))

        documents = loader.load()

        # Replace the temporary path with the user's original filename.
        for document in documents:
            document.metadata["source"] = filename
            document.metadata["file_type"] = extension.lstrip(".")

        return [
            document
            for document in documents
            if document.page_content.strip()
        ]

    finally:
        temp_path.unlink(missing_ok=True)