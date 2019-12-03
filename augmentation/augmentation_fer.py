from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os 


data_path = '/Data2/home/intern/exhibition/temp_hw/aug/'

folder_list = os.listdir(data_path)

for idx, folder in enumerate(folder_list[:]):

    print(idx)

    folder_path = os.path.join(data_path, folder)

    file_list = os.listdir(folder_path)


    for idx, file in enumerate(file_list[:]):



        datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                brightness_range=[0.5,1.0],
                fill_mode='nearest')


        img = load_img(os.path.join(folder_path, file))

        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0

        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=folder_path, save_prefix=file, save_format='jpeg'):
            i += 1
            if i >= 9:
                break
