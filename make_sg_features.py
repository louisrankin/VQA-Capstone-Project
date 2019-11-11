# Import libraries for inspecting files
import json
import pickle
import h5py
import numpy as np

import time

word_embed_size = 50
max_objects = 100

# Get gloVe embeddings and create lookup dictionary
print('\nCreating lookup dictionary for embeddings...')
embed_dict = {}
start_time = time.time()
with open('./data/glove.6B.'+str(word_embed_size)+'d.txt') as f:
    for line in f:
        wordsAndVectors = line.strip().split(' ')
        word = wordsAndVectors[0]
        vector = wordsAndVectors[1:]
        vector = list(map(float, vector))
        embed_dict[word] = vector

print('Time elapsed:', np.round(time.time() - start_time,1), "seconds")

# Function that breaks text into words and uses the embedding dictionary returning the average for each word
def encode_text(text):
    word_list = list(text.split())
    if len(word_list) == 1:
        if word_list[0] in embed_dict:

            embedding = np.array(embed_dict[word_list[0]])
        else:

            embedding = np.zeros(word_embed_size)
    else:
        embed_list = np.zeros((len(word_list), word_embed_size))

        for i in range(len(word_list)):
            if word_list[i] in embed_dict:
                embed_list[i] = embed_dict[word_list[i]]

        embedding = np.mean(embed_list, axis=0)
    return embedding


def parse_graph (data_sg, num_images_feed, index_start=0, report_interval=5000):

    # Initialise data files
    bboxes_matrix = np.zeros((num_images_feed, max_objects, 4))
    features_matrix = np.zeros((num_images_feed, max_objects, 4*word_embed_size))
    image_info_dict = {}
    image_count=0
    #index_to_image = {} #Z# inspection code

    start_time = time.time()

    for image_id in data_sg:

        if image_count % report_interval == 0:
            print('Progress:', str(np.round(100*(image_count / num_images_feed),0))
                  +'%, Time:', np.round(time.time() - start_time,1), "seconds")

        object_name_dict = {}

        # First pass is to go through the object list and build dictionary of objects so relations can recognise
        for object_id in data_sg[image_id]['objects']:
            object_name_dict[object_id] = data_sg[image_id]['objects'][object_id]['name']

        # Second pass is to build the embeddings
        object_count = 0

        #obj_list = [] #Z
        for object_id in data_sg[image_id]['objects']:

            # Get bounding box details and store
            x = data_sg[image_id]['objects'][object_id]['x']
            y = data_sg[image_id]['objects'][object_id]['y']
            h = data_sg[image_id]['objects'][object_id]['h']
            w = data_sg[image_id]['objects'][object_id]['w']
            bboxes_matrix[image_count, object_count] = [x,y,x+w,y+h]

            obj_name = data_sg[image_id]['objects'][object_id]['name']
            #obj_list.append(obj_name) #Z
            object_name_encode = encode_text(obj_name)
            objects_list = object_name_encode

            # Get mean attribute encodings for each object
            n_attribs = len(data_sg[image_id]['objects'][object_id]['attributes'])
            attribs_list = np.zeros(word_embed_size)

            if n_attribs >= 1:
                if n_attribs == 1:
                    for attribute in data_sg[image_id]['objects'][object_id]['attributes']:
                        attribs_list = encode_text(attribute)
                elif n_attribs > 1:
                    attribs_sublists = np.zeros((n_attribs, word_embed_size))
                    attrib_count = 0
                    for attribute in data_sg[image_id]['objects'][object_id]['attributes']:

                        attribs_sublists[attrib_count] = encode_text(attribute)
                        attrib_count += 1
                    attribs_list = np.mean(attribs_sublists, axis=0)

            # Get mean relation encodings for each object
            n_relations = len(data_sg[image_id]['objects'][object_id]['relations'])
            relation_object_list = np.zeros(word_embed_size)
            relationship_list = np.zeros(word_embed_size)

            if n_relations >= 1:
                if n_relations == 1:

                    for relation in data_sg[image_id]['objects'][object_id]['relations']:
                        rel_object_id = relation['object']
                        if rel_object_id in object_name_dict:
                            rel_object_name = object_name_dict[rel_object_id]
                            relations_object_list = encode_text(rel_object_name)
                        relationship_list = encode_text(relation['name'])

                elif n_relations > 1:
                    relations_objects_sublists = np.zeros((n_relations, word_embed_size))
                    relationships_sublists = np.zeros((n_relations, word_embed_size))
                    relations_count = 0
                    for relation in data_sg[image_id]['objects'][object_id]['relations']:
                        rel_object_id = relation['object']
                        if rel_object_id in object_name_dict:
                            rel_object_name = object_name_dict[rel_object_id]
                            relations_objects_sublists[relations_count] = encode_text(rel_object_name)
                        relationships_sublists[relations_count] = encode_text(relation['name'])
                        relations_count += 1
                    relation_object_list = np.mean(relations_objects_sublists, axis=0)
                    relationship_list = np.mean(relationships_sublists, axis=0)

            features_list = np.concatenate([objects_list, attribs_list, relation_object_list, relationship_list])
            features_matrix[image_count, object_count] = features_list

            object_count += 1
            if object_count >= 100:
                break

        # Create image summary info
        image_info_dict[str(image_id)] = {'width': data_sg[image_id]['width'], 'objectsNum': object_count,
                                          'height': data_sg[image_id]['height'], 'index': image_count+index_start}

        image_count +=1
        if image_count >= num_images_feed:
            break

    print('Action complete. Time elapsed:', np.round(time.time() - start_time,0), "seconds")
    return bboxes_matrix, features_matrix, image_info_dict, image_count

# Load training scene graph files
print('\nLoading training scene graph file and generating object features...')
with open('./sceneGraphs/train_sceneGraphs.json') as f:
    data_sg = json.load(f)

bboxes_train, features_train, image_info_train, n_train = parse_graph (data_sg, len(data_sg))

print('Number of image items:', len(image_info_train))

# Load validation scene graph files
print('\nLoading validation scene graph file and generating object features...')
with open('./sceneGraphs/val_sceneGraphs.json') as f:
    data_sg = json.load(f)

bboxes_val, features_val, image_info_val, n_val = parse_graph (data_sg, len(data_sg), index_start = n_train)

print('Number of image items:', len(image_info_val))

# Concatenate data
print('\nConcatenating training and validation feature data...')
bboxes_matrix = np.concatenate((bboxes_train, bboxes_val))
features_matrix = np.concatenate((features_train, features_val))
image_info_dict = {**image_info_train, **image_info_val}

bboxes_train = bboxes_val = features_train = features_val = image_info_train = image_info_val = data_sg = None

# Check question and answer code for images not in scene sceneGraphs
print('\nConfirm if question and answer code has no matching scene graph...')
with open('./data/balanced_train_data.json') as f:
    data_bal_train = json.load(f)

bal_tr_key = list(data_bal_train.keys())[0]
print('Number of question instances:', len(data_bal_train['questions']))

i = 0
i_max = 999000
data_bal_train_sg = {}
n_false = 0

for qa_item in data_bal_train['questions']:
    #print(qa_item['imageId'], qa_item)
    if qa_item['imageId'] in image_info_dict:
        #print(True)
        n_false = n_false
    else:
        #print(False)
        n_false += 1
    i += 1
    if i > i_max:
        break
print("Number not matching =", n_false)

# Saving data to JSON and H5 formats
print('\nWriting summary JSON File...')
with open('./data/gqa_objects_sg_merged_info.json', 'w') as outfile:
    json.dump(image_info_dict, outfile)

# Write H5PY File
print('\nWriting bboxes and features H5 File...')
export_file = h5py.File('./data/gqa_objects_sg.h5', 'w')
export_file.create_dataset('bboxes', data=bboxes_matrix)
export_file.create_dataset('features', data=features_matrix)
export_file.close()
print('\nOperation complete.\n')
