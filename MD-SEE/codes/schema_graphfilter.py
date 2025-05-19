import os
import json
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from FlagEmbedding import BGEM3FlagModel

def save_embeddings(embeddings, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings saved to {filepath}")

def schema_to_text(schema):
    args = ' '.join(f"{role}: {desc}" for role, desc in schema.get('arguments', {}).items())
    return f"{schema.get('name', '')} {args} {schema.get('description', '')}"

def print_top_similar_pairs(similarity_matrix, schema_embeddings, top_n, phase, iteration):
    upper_tri_indices = np.triu_indices(len(schema_embeddings), k=1)
    similarities = similarity_matrix[upper_tri_indices]
    pairs = sorted(zip(similarities, zip(*upper_tri_indices)), reverse=True, key=lambda x: x[0])[:50]
    count = 0
    print(f"\nTop {top_n} similar schema pairs across datasets {phase} iteration {iteration}:")
    for sim_score, (idx1, idx2) in pairs:
        schema1, schema2 = schema_embeddings[idx1]['schema'], schema_embeddings[idx2]['schema']
        if schema1['dataset'] != schema2['dataset']:
            print(f"\nPair {count + 1}: Similarity Score: {sim_score:.4f}")
            print(f"Schema 1: {schema1['name']} (ID: {schema1.get('id', 'N/A')}, Dataset: {schema1['dataset']})")
            print(f"Arguments: {', '.join(schema1['arguments'].keys())}")
            print(f"Schema 2: {schema2['name']} (ID: {schema2.get('id', 'N/A')}, Dataset: {schema2['dataset']})")
            print(f"Arguments: {', '.join(schema2['arguments'].keys())}")
            count += 1
        if count >= 10:
            break

def greedy_maximum_independent_set(adj_list):
    remaining = set(adj_list.keys())
    independent_set = set()
    degrees = {node: len(neighbors) for node, neighbors in adj_list.items()}

    while remaining:
        node = min(remaining, key=lambda x: degrees.get(x, 0))
        independent_set.add(node)
        remaining.remove(node)
        neighbors = adj_list[node]
        remaining.difference_update(neighbors)
        for neighbor in neighbors:
            degrees.pop(neighbor, None)
            for n in adj_list.get(neighbor, []):
                if n in remaining:
                    degrees[n] -= 1
            degrees.pop(node, None)

    return independent_set

def filter_schemas_by_similarity(schema_embeddings, threshold):
    print("\nComputing pairwise similarities for filtering...")
    embeddings_array = np.vstack([item['embedding'] for item in schema_embeddings])
    similarity_matrix = cosine_similarity(embeddings_array)
    print("Building the similarity graph...")

    adj_list = {i: {j for j in range(len(schema_embeddings)) if i != j and similarity_matrix[i, j] >= threshold} for i in range(len(schema_embeddings))}
    print(f"Graph constructed with {len(schema_embeddings)} nodes.")

    print("Finding a large independent set to maximize schema retention...")
    retained_indices = greedy_maximum_independent_set(adj_list)
    print(f"Independent set found. Retained {len(retained_indices)} schemas.")

    filtered_embeddings = [schema_embeddings[i] for i in retained_indices]
    return filtered_embeddings

def load_filtered_schemas(filtered_schema_file):
    """Load filtered schemas and create a mapping of schema identifiers to schema data."""
    filtered_schemas = {}
    with open(filtered_schema_file, 'r', encoding='utf-8') as f:
        for line in f:
            schema = json.loads(line.strip())
            name = schema.get('name', '')
            dataset = schema.get('dataset', '').strip().lower()
            schema_identifier = (name, dataset)
            filtered_schemas[schema_identifier] = schema  # Map to the schema data
    return filtered_schemas

def filter_and_annotate_data_instances(input_file, output_file, filtered_schemas):
    """Filter data instances based on the filtered schemas and keep the original schema_id."""
    kept_instances = 0
    total_instances = 0
    dataset_instance_counts = {}  # Counts per dataset
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            total_instances += 1
            data_instance = json.loads(line.strip())
            labels = data_instance.get('label', [])
            dataset = data_instance.get('dataset', 'unknown').strip().lower()
            new_labels = []
            for event in labels:
                name = event.get('name', '')
                schema_identifier = (name, dataset)
                if schema_identifier in filtered_schemas:
                    new_labels.append(event)
                # Else, do not include this event
            if new_labels:
                data_instance['label'] = new_labels
                json.dump(data_instance, f_out, ensure_ascii=False)
                f_out.write('\n')
                kept_instances += 1
                # Count instances per dataset
                dataset_instance_counts[dataset] = dataset_instance_counts.get(dataset, 0) + 1
            # Else, discard the data instance
    print(f"Processed {total_instances} instances in {input_file}. Kept {kept_instances} instances.")
    print(f"Instance counts per dataset in {os.path.basename(output_file)}:")
    for dataset_name, count in dataset_instance_counts.items():
        print(f"{dataset_name.capitalize()}: {count}")
    return kept_instances, total_instances, dataset_instance_counts

def main(similarity_threshold=0.80):
    current_dir = ''
    input_file = os.path.join(current_dir, '')
    output_file = os.path.join(current_dir, '')
    embeddings_file = os.path.join(current_dir, 'schema_embeddings.pkl')

    model = BGEM3FlagModel('/models/bgem3', use_fp16=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        schemas = [json.loads(line.strip()) for line in f]

    dataset_counts = {}
    for schema in schemas:
        dataset = schema['dataset']
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
    print("Number of schemas per dataset (before filtering):")
    for dataset, count in dataset_counts.items():
        print(f"{dataset.capitalize()}: {count}")

    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f_emb:
            schema_embeddings = pickle.load(f_emb)
        print(f"Loaded {len(schema_embeddings)} embeddings.")
    else:
        print("\nEncoding schemas and saving embeddings...")
        schema_embeddings = []
        for idx, schema in enumerate(schemas):
            sentence = schema_to_text(schema)
            output = model.encode([sentence], return_dense=True, return_colbert_vecs=False)
            schema_embeddings.append({'schema': schema, 'embedding': output['dense_vecs'][0]})
            if (idx + 1) % 100 == 0 or (idx + 1) == len(schemas):
                print(f"Encoded {idx + 1}/{len(schemas)} schemas.")
        save_embeddings(schema_embeddings, embeddings_file)

    print("\nComputing pairwise similarities before filtering...")
    embeddings_array = np.vstack([item['embedding'] for item in schema_embeddings])
    similarity_matrix = cosine_similarity(embeddings_array)
    print_top_similar_pairs(similarity_matrix, schema_embeddings, top_n=30, phase="before", iteration=0)

    filtered_schema_embeddings = filter_schemas_by_similarity(schema_embeddings, similarity_threshold)

    print("\nComputing pairwise similarities after filtering...")
    filtered_embeddings_array = np.vstack([item['embedding'] for item in filtered_schema_embeddings])
    filtered_similarity_matrix = cosine_similarity(filtered_embeddings_array)
    print_top_similar_pairs(filtered_similarity_matrix, filtered_schema_embeddings, top_n=10, phase="after", iteration=1)

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in filtered_schema_embeddings:
            json.dump(item['schema'], f, ensure_ascii=False)
            f.write('\n')

    # Load the filtered schemas
    filtered_schemas = {}
    for item in filtered_schema_embeddings:
        schema = item['schema']
        name = schema.get('name', '')
        dataset = schema.get('dataset', '').strip().lower()
        schema_identifier = (name, dataset)
        filtered_schemas[schema_identifier] = schema

    # Filter and annotate data instances
    total_dataset_counts = {}
    total_instances_counts = {}
    data_files = {
        'train': '',
        'dev': '',
        'test': ''
    }

    for split, data_file in data_files.items():
        filtered_file = data_file.replace('.json', '_filtered.json')
        print(f"\nProcessing {data_file}...")
        kept_instances, total_instances, dataset_instance_counts = filter_and_annotate_data_instances(data_file, filtered_file, filtered_schemas)
        total_dataset_counts[split] = dataset_instance_counts
        total_instances_counts[split] = {'processed': total_instances, 'kept': kept_instances}

    dataset_counts_after = {}
    for item in filtered_schema_embeddings:
        schema = item['schema']
        dataset = schema['dataset']
        dataset_counts_after[dataset] = dataset_counts_after.get(dataset, 0) + 1
    print("\nNumber of schemas per dataset (after filtering):")
    for dataset, count in dataset_counts_after.items():
        print(f"{dataset.capitalize()}: {count}")
    print(f"Filtering complete. {len(filtered_schema_embeddings)} unique schemas saved to {output_file}.")

    # Print counts of data instances per dataset after filtering
    print("\nData instance counts per dataset after filtering:")
    for split, data_file in data_files.items():
        print(f"\nSplit: {split.capitalize()}")
        dataset_instance_counts = total_dataset_counts[split]
        total_kept_instances = total_instances_counts[split]['kept']
        total_processed_instances = total_instances_counts[split]['processed']
        print(f"Total instances processed: {total_processed_instances}")
        print(f"Total instances kept: {total_kept_instances}")
        for dataset_name, count in dataset_instance_counts.items():
            print(f"{dataset_name.capitalize()}: {count}")

if __name__ == "__main__":
    main(similarity_threshold=0.85)