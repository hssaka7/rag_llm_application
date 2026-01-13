
from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever


from langchain_community.query_constructors.milvus import MilvusTranslator



metadata_field_info = [
    AttributeInfo(
        name="company",
        description="The company mentioned in the document",
        type="string",
    ),
    AttributeInfo(
        name="sentiment",
        description="Sentiment of the document",
        type="string",  # e.g. positive, neutral, negative
    ),
    AttributeInfo(
        name="year",
        description="Year the document was published",
        type="integer",
    ),
    AttributeInfo(
        name="source",
        description="Source of the news article",
        type="string",
    ),
]



print("hello")