from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, Conv2DTranspose, MaxPooling2D, ZeroPadding2D
from keras import backend as keras

class modelsClass(object):

    def __init__(self, img_rows = 272, img_cols = 480):

        self.img_rows = img_rows
        self.img_cols = img_cols

    def addPadding(self, layer, level): #height, width, level):
    
        w1, h1 = self.img_cols, self.img_rows
        w2, h2 = int(w1/2), int(h1/2)
        w3, h3 = int(w2/2), int(h2/2)
        w4, h4 = int(w3/2), int(h3/2)
        h = [h1,h2,h3,h4]
        w = [w1,w2,w3,w4]
        
        # Target width and height
        tw = w[level-1]
        th = h[level-1]
        
        # Source width and height
        lsize = keras.int_shape(layer)
        sh = lsize[1]
        sw = lsize[2]

        pw = (0, tw - sw)
        ph = (0, th - sh)

        layer = ZeroPadding2D(padding=(ph,pw),data_format="channels_last")(layer)
    
        return layer
        
    def getDeepGyro(self):

        input_blurred = Input((self.img_rows, self.img_cols,3))
        input_blurx = Input((self.img_rows, self.img_cols,1))
        input_blury = Input((self.img_rows, self.img_cols,1))
        
        inputs = concatenate([input_blurred,input_blurx,input_blury])
        
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

        up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
        up6 = self.addPadding(up6,level=4)
        up6 = concatenate([up6,conv4], axis=3)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

        up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
        up7 = self.addPadding(up7,level=3)
        up7 = concatenate([up7,conv3], axis=3)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

        up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
        up8 = self.addPadding(up8,level=2)
        up8 = concatenate([up8,conv2], axis=3)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

        up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
        up9 = self.addPadding(up9,level=1)
        up9 = concatenate([up9,conv1], axis=3)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(3, (1, 1), activation='linear')(conv9)
        
        model = Model(inputs=[input_blurred,input_blurx,input_blury], outputs=conv10)
           
        #adam = optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        #model.compile(optimizer = adam, loss = 'mean_squared_error')

        return model
        
    def getDeepBlind(self):

        input_blurred = Input((self.img_rows, self.img_cols,3))
        
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_blurred)
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

        up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
        up6 = self.addPadding(up6,level=4)
        up6 = concatenate([up6,conv4], axis=3)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

        up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
        up7 = self.addPadding(up7,level=3)
        up7 = concatenate([up7,conv3], axis=3)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

        up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
        up8 = self.addPadding(up8,level=2)
        up8 = concatenate([up8,conv2], axis=3)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

        up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
        up9 = self.addPadding(up9,level=1)
        up9 = concatenate([up9,conv1], axis=3)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(3, (1, 1), activation='linear')(conv9)
        
        model = Model(inputs=input_blurred, outputs=conv10)
        
        #adam = optimizers.Adam(lr=0.0000125, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        #model.compile(optimizer = adam, loss = 'mean_squared_error')

        return model
