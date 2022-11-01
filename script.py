import numpy as np
import os
import cv2

FOLDER_DIR = '' # relatively trustworthy dir
NUM_CLASSES = 10
DATA_BATCHES= ['cifar-10-batches-py/data_batch_1',
               'cifar-10-batches-py/data_batch_2',
               'cifar-10-batches-py/data_batch_3',
               'cifar-10-batches-py/data_batch_4',
               'cifar-10-batches-py/data_batch_5']
VALIDATION_TEST_BATCH = 'cifar-10-batches-py/test_batch'
VALIDATION_RATE = 0.5

# extract file function from batches
def unpickle(file):
    import _pickle as cpickle
    with open(file, 'rb') as fo:
        data = cpickle.load(fo,  encoding='latin1')
    return data

if __name__ == '__main__':
  training_labels = np.array([]) 
  training_images = np.array([]) 
  for data_batch in DATA_BATCHES : 
    dic = unpickle(data_batch)
    training_labels = np.append(training_labels, dic['labels'])
    data = np.array(dic['data']) 
    dic['data'] = np.reshape(data, (10000, 3, 32, 32)).transpose([0, 2 , 3, 1 ])
    if training_images.size == 0 : 
       training_images = dic['data']
    else : 
      training_images = np.append(training_images, dic['data'], axis=0)

  training_labels = np.reshape(training_labels, (-1, 1))

  dic = unpickle(VALIDATION_TEST_BATCH)
  dic['data'] = np.array(np.reshape(np.array(dic['data']) , (10000, 3, 32, 32)))
  dic['data'] = dic['data'].transpose([0, 2 , 3, 1 ])

  len = int(VALIDATION_RATE * 10000) 
  validation_labels = np.array(dic['labels'][:len]).reshape((-1,1))
  validation_images = np.array(dic['data'][:len])


  testing_labels = np.array(dic['labels'][len:]).reshape((-1,1))
  testing_images = np.array(dic['data'][len:])



  # change the ori_label for the name set
  ori_label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  folders = ['train', 'validate', 'test']
  parent_dir = 'cifar-10'



  for folder in folders : 
    dir = os.path.join(parent_dir, folder)
    os.mkdir(dir)
  for folder in folders : 
    for label in ori_label : 
      dir = os.path.join(parent_dir + '/' + folder, label)
      os.mkdir(dir)


  def generate_image(data, labels, folder_name) : 
    count_label = np.zeros(NUM_CLASSES)
    for j, i in enumerate(labels):
      index = int(i) 
      count_label[index] += 1
      print(j,folder_name)
      img_dir = '{}/{}/{}/{}.jpg'.format(parent_dir, folder_name, ori_label[index], int(count_label[index]))
      # print(img_dir)
      cv2.imwrite(img_dir, data[j])


  generate_image(training_images, training_labels, folders[0])
  generate_image(validation_images, validation_labels, folders[1])
  generate_image(testing_images, testing_labels, folders[2])
