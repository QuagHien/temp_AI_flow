from vector_store import * 
from embeding import * 
import argparse
from datasets import load_dataset, DatasetDict, concatenate_datasets
import os

def main():
    # Argument
    parser = argparse.ArgumentParser(description="Read and create database", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--collection_name", type=str, required=True, help="Name of the collection")
    parser.add_argument("--similarity_metric", type=str, default="cosine", choices=["cosine", "euclidean", "dot", "manhattan"], help="Similarity metric for collection")
    parser.add_argument("--collection_type", type=str, default="image-text", choices=["image", "text", "image-text"], help="Type of the collection")
    parser.add_argument("--dataset_name", nargs="+", type=str, required=True, help="List of the dataset name")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory for caching dataset")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes for loading dataset")
    parser.add_argument("--batch_size", type=int, default=3, help="Batch size for processing dataset")
    args = parser.parse_args()

    # Create collection and dadatabase dir
    os.makedirs("result", exist_ok=True)
    os.makedirs("result/database", exist_ok=True)
    os.makedirs("result/collection", exist_ok=True)

    # Initialize
    vector_store = QdrantVectorSpace(
        collection_name=args.collection_name,
        similarity_metric=args.similarity_metric,
        collection_type=args.collection_type
    )
    model = JinaImageEmbeding('weights/jina-clip-v1')
    image_database = ImageDatabaseSQL('result/database/image.sqlite', 'image')
    image_database.connect()
    image_database.create_table()

    # Load dataset
    ds = []
    print("Loading: ")
    for each_dataset in args.dataset_name:
        print(f"===> Loading dataset: {each_dataset}")
        ds_each = load_dataset(
            each_dataset,
            cache_dir=args.cache_dir
        )
        ds_each['train'] = ds_each['train'].remove_columns(
            [col for col in ds_each['train'].column_names if col not in ['image', 'video_id', 'frame_id']]
        )
        ds.append(ds_each['train'])
    summary_ds = concatenate_datasets(ds)

    # Embedding
    def embedding(x):
        result = {
            'video_id': x['video_id'],
            'frame_id': x['frame_id'],
            'image_embedding': [],
            'text_embedding': []        
        }
        if args.collection_type in ["image-text", "image"]:
            result['image_embedding'] = model.inference(x['image'], norm_embeds=True)
        else:
            result['image_embedding'] = [["a" for _ in range(768)] for _ in range(len(x['video_id']))]

        if args.collection_type in ["image-text", "text"]:
            result['text_embedding'] = [[0 for _ in range(768)] for _ in range(len(x['video_id']))]
        else:
            result['text_embedding'] = [["a" for _ in range(768)] for _ in range(len(x['video_id']))]
        return result

    # Add database
    def database_add(x):
        image_database.insert_multiple(x)

    # Add to database
    print("Adding to database:")
    add_database = summary_ds.map(
        database_add, 
        batched=True,
        batch_size=1000,
        num_proc=1
    )

    # Add pairs to the vector space
    print("Embedding dataset:")
    processed_ds = summary_ds.map(
        embedding, 
        batched=True, 
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=['image']
    )
    vector_store.add_pairs(processed_ds)

    # # Create snapshot
    vector_store.create_snapshot('result/collection')

if __name__ == "__main__":
    main()