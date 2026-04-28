from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # filter
    from .generate.kbc_chunk_generator import KBCChunkGenerator
    from .generate.kbc_chunk_generator_batch import KBCChunkGeneratorBatch
    from .generate.mineru_operators import FileOrURLToMarkdownConverterLocal
    from .generate.mineru_operators import FileOrURLToMarkdownConverterAPI
    from .generate.mineru_operators import FileOrURLToMarkdownConverterFlash
    from .generate.kbc_text_cleaner import KBCTextCleaner
    from .generate.kbc_text_cleaner_batch import KBCTextCleanerBatch

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/pdf2text/"


    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/pdf2text/", _import_structure)
