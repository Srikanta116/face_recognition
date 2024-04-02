# import os

# import face_recognition

# from collections import defaultdict

# import time

# start = time.time()

# def find_person_in_groups(person_image_path, group_images_folder):

#     person_image = face_recognition.load_image_file(person_image_path)

#     person_encoding = face_recognition.face_encodings(person_image)[0]

#     # Preload group image encodings

#     group_encodings_dict = {}

#     for group_image_file in os.listdir(group_images_folder):

#         group_image_path = os.path.join(group_images_folder, group_image_file)

#         group_image = face_recognition.load_image_file(group_image_path)

#         group_encodings_dict[group_image_path] = face_recognition.face_encodings(group_image)

#     matched_groups = defaultdict(list)

#     for group_image_path, group_encodings in group_encodings_dict.items():

#         for group_encoding in group_encodings:

#             matches = face_recognition.compare_faces([person_encoding], group_encoding)

#             if any(matches):

#                 matched_groups[group_image_path].append(person_encoding)

#                 break # Stop comparing if a match is found

#     return matched_groups

# person_image_path = "facepic/steve.jpg"

# group_images_folder = "images"

# matched_groups = find_person_in_groups(person_image_path, group_images_folder)

# timess = time.time() - start

# for group_image, matched_persons in matched_groups.items():

#     print(f"Person found in {group_image}: {len(matched_persons)} times.")

# print("Time taken: ", timess)

import os
import numpy
import face_recognition
from collections import defaultdict
import time
import multiprocessing
import pickle
from PIL import Image

def encode_image(image_path):
    image = Image.open(image_path)
    image = image.resize((800, 600)) 
    image = numpy.array(image)

    return face_recognition.face_encodings(image)




def create_person_data(person_image_path):
    person_image = face_recognition.load_image_file(person_image_path)
    person_encoding = face_recognition.face_encodings(person_image)[0]
    return {"encoding": person_encoding, "images": [person_image_path]}

def find_person_in_groups(person_image_path, group_images_folder):
    person_image = face_recognition.load_image_file(person_image_path)
    person_encoding = face_recognition.face_encodings(person_image)[0]

    if os.path.exists("encoding_database.pkl"):
        with open("encoding_database.pkl", "rb") as f:
            encoding_database = pickle.load(f)
    else:
        encoding_database = {}

    person_match = None
    for person, data in encoding_database.items():
        if data is not None and face_recognition.compare_faces([data["encoding"]], person_encoding)[0]:
            person_match = person
            break


    if person_match:
        # print("Person already exists")
        person_data = encoding_database[person_match]
        existing_images = person_data["images"]
        new_images = [os.path.join(group_images_folder, img) for img in os.listdir(group_images_folder) if img not in existing_images]

    else:
        encoding_database[person_image_path] = create_person_data(person_image_path)
        new_images = [os.path.join(group_images_folder, img) for img in os.listdir(group_images_folder)]

    group_encodings_dict = {}
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    for group_image_path in new_images:
        group_encodings_dict[group_image_path] = pool.apply_async(encode_image, (group_image_path,))

    pool.close()
    pool.join()

    matched_groups = defaultdict(list)

    for group_image_path, group_encodings_async in group_encodings_dict.items():
        group_encodings = group_encodings_async.get()

        for group_encoding in group_encodings:
            matches = face_recognition.compare_faces([person_encoding], group_encoding)
            if any(matches):
                matched_groups[group_image_path].append(person_encoding)
                break  

    if person_match:
        encoding_database[person_match]["images"].extend(new_images)
    with open("encoding_database.pkl", "wb") as f:
        pickle.dump(encoding_database, f)

    return matched_groups

if __name__ == '__main__':
    start = time.time()
    person_image_path = "facepic/stark.jpg"
    group_images_folder = "images"

    matched_groups = find_person_in_groups(person_image_path, group_images_folder)

    timess = time.time() - start

    if matched_groups:
        for group_image in matched_groups.keys():
            print(f"Person found in {group_image}")
    else:
        print("Person not found in any image")

    print("Time taken: ", timess)


# import pickle

# with open('encoding_database.pkl', 'rb') as f:
#     data = pickle.load(f)

# print(data)