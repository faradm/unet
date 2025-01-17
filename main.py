from model import *
from data import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.05,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(4,'/content/data/train','image','label',data_gen_args,save_to_dir = None)
model = unet()
model_checkpoint = ModelCheckpoint('unet_breast.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=300,epochs=10,callbacks=[model_checkpoint])

testGene = testGenerator("/content/data/test/image")
results = model.predict_generator(testGene,30,verbose=1)
saveResult("/cotent/gdrive/My Drive/unet/result",results)
