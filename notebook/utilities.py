#----------------------------------------------------------------------- NEEDED PACKAGES ---------------------------------------------------------------------

import pandas as pd
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sn

import tensorflow as tf


#-------------------------------------------------------------------- PREPROCESSING ------------------------------------------------------------------------

def emoticon_replacer(sentence):
    """
    replace emoticons in a text with words with similiar semantic value
    """
    sentence=re.sub(r'😍|🥰|😘|😻|❤️|🧡|💛|💚|💙|💜|🖤|🤍|🤎|💕|💞|💓|💗|💖|💘|💝', 'adoro ', sentence)
    sentence=re.sub(r'😀|😃|😄|😁|😆|😂|🤣|😹|😸', 'ahah ', sentence)
    sentence=re.sub(r'😡|🤬|👿|🤡', 'infame ', sentence)
    sentence=re.sub(r'✈️|🔥|💫|⭐️|🌟|✨|💥|🛫|🛬|🛩|🚀', 'wow ', sentence)
    sentence=re.sub(r'😢|😭|😿', 'piango ', sentence)
    sentence=re.sub(r'🤢|🤮', 'schifo ', sentence)    
    return sentence


def tokenize(sentence, tokenizer, SEQ_LEN=50, emotic=False):
    """
    tokenizes a sentence preparing it for bert model
    """
    sentence=re.sub(r'(http\S+)|(\s)#\w+', '', sentence)
    if emotic==True:
        sentence=emoticon_replacer(sentence)
    tokens = tokenizer.encode_plus(sentence, max_length=SEQ_LEN,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True,
                                   return_token_type_ids=False, return_tensors='tf')
    return tokens['input_ids'], tokens['attention_mask']

def preprocess_txt(input_ids, masks, labels):
    """
    format for tf model
    """
    return {'input_ids': input_ids, 'attention_mask': masks}, labels

def preprocess_txt_imtxt(input_ids, masks, input_ids2, masks2, labels):
    """
    format for tf model
    """
    return {'input_ids': input_ids, 'attention_mask': masks, 
            'input_ids2': input_ids2, 'attention_mask2': masks2}, labels

def aug(image, label):
    """
    perform data augumentation
    """
    IMG_SIZE=224
    image = tf.image.random_brightness(image, 0.15)
    image = tf.image.random_contrast(image, upper=1.5, lower=0.5)
    image = tf.image.random_saturation(image,upper=1.5, lower=0.5)
    image = tf.image.random_hue(image, 0.15)
    #if tf.random.uniform([])>0.5:
        #image= tf.image.flip_left_right(image)
    if tf.random.uniform([])>0:
        image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6) 
   # Random crop back to the original size
        image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
    image = tf.clip_by_value(image, 0, 1)
    return image, label


def preprocess_image(image, labels, prediction=False):
    """
    format for tf model + open image and prepare it for resnet
    """
    if prediction==False:
        image = tf.io.read_file(image)
        image = tf.io.decode_image(image, channels=3,expand_animations = False)
    image = tf.cast(image, tf.float32) / 255.0 
    image = tf.image.resize(image, size=(224, 224))
    
    if prediction==True:
        image= tf.expand_dims(image, axis=0)
        return image
   
    return image, labels

def preprocess_txt_image(input_ids, masks, path, labels):
    """
    format for tf model + open image and prepare it for resnet
    """
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3,expand_animations = False)
    image = tf.cast(image, tf.float32) / 255.0 
    image = tf.image.resize(image, size=(224, 224)) 
    return {'input_ids': input_ids, 'attention_mask': masks, 'images': image}, labels

def preprocess_txt_imtxt_image(input_ids, masks, input_ids2, masks2, path, labels):
    """
    format for tf model + open image and prepare it for resnet
    """
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3,expand_animations = False)
    image = tf.cast(image, tf.float32) / 255.0 
    image = tf.image.resize(image, size=(224, 224)) 
    return {'input_ids': input_ids, 'attention_mask': masks, 
            'input_ids2': input_ids2, 'attention_mask2': masks2,
            'images': image}, labels


#------------------------------------------------------     DATA IMPORTER     ---------------------------------------------------------------------


def df_to_tf_data(df, tokenizer, txt=True, image=False, imtxt=False, 
                  SEQ_LEN=50, SEQ_LEN2=10, 
                  shuffle=True, emotic=False, augmentation=False):
    """
    from dataframe to tensorflow Dataset object
    SEQ_LEN: max number of tokens from text before truncation
    SEQ_LEN2: max number of tokens from in image text before truncation
    emoticon: wheater or not translate emoticons
    augumentation: wheater or not to augment images
    """ 
    df=df.replace({'negative': 0,'neutral':1, 'positive': 2})
    arr = df['sentiment'].values  # take sentiment column in df as array
    labels = np.zeros((arr.size, arr.max()+1))  # initialize empty (all zero) label array
    labels[np.arange(arr.size), arr] = 1  # add ones in indices where we have a value
    
    if txt == True:
        Xids = np.zeros((len(df), SEQ_LEN))
        Xmask = np.zeros((len(df), SEQ_LEN))
        for i, sentence in enumerate(df['text']):
            Xids[i, :], Xmask[i, :] = tokenize(sentence, tokenizer, SEQ_LEN, emotic=emotic)
                
    if imtxt == True:
        Xids2 = np.zeros((len(df), SEQ_LEN2))
        Xmask2 = np.zeros((len(df), SEQ_LEN2))
        for i, sentence in enumerate(df['inimagetext']):
            Xids2[i, :], Xmask2[i, :] = tokenize(sentence, tokenizer, SEQ_LEN2, emotic=False)

    if image == True:        
        paths = []
        for i, img in enumerate(df['path']):
            paths.append(img)
        
        
    if txt==True:
        if imtxt == True:
            if image == True:
                dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, Xids2, Xmask2, paths, labels))
                preprocess=preprocess_txt_imtxt_image
            else:
                dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, Xids2, Xmask2, labels))
                preprocess=preprocess_txt_imtxt
        else:
            if image == True:
                dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, paths, labels))
                preprocess=preprocess_txt_image
            else:
                dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))
                preprocess=preprocess_txt
    else:
        if image == True:
            dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
            preprocess=preprocess_image

    # shuffle and batch the dataset
    if shuffle==True:
        if augmentation==True:
            dataset = dataset.shuffle(5000).map(preprocess).map(aug).batch(4)
        else:
            dataset = dataset.shuffle(5000).map(preprocess).batch(4)
    else:
        dataset = dataset.map(preprocess).batch(4)
    
    return(dataset)

#------------------------------------------------------     PLOTS     --------------------------------------------------------------------------


def plot_history(history, title, save=False):
    f = plt.figure(figsize=(15,4))
    plt.suptitle(title)
    f.add_subplot(1,2, 1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.legend(['Train loss', 'Val loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    f.add_subplot(1,2, 2)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.legend(['Train Accuracy', 'Val accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    if save!=False:
        f.savefig('C:\\Users\\Egon\\Desktop\\tesi\\immagini tesi\\modelli\\'+title+'.png')            

        
        
def confusion_matrix_plotter(true,pred, title='confusion_matrix', normalize='true', save=False):

    lab=['negative', 'neutral', 'positive']    
    mat=np.array(tf.math.confusion_matrix(predictions=pred, labels=true))
    
    if normalize=='true':
        mat=mat/mat.sum(axis=1)
    
    fig, ax= plt.subplots(figsize = (15,8))
    
    sn.heatmap(pd.DataFrame(mat, index=lab, columns=lab), annot=True, cmap='flare', annot_kws={"size": 20})
    plt.title(title, fontdict={'color':'DarkRed', 'size':30})
    ax.set_yticklabels(labels=lab,va='center', fontdict={'size':15})
    ax.set_xticklabels(labels=lab, fontdict={'size':15})
    ax.set_xlabel('Predicted labels',fontdict={'color':'DarkRed', 'size':20})
    ax.set_ylabel('True labels',fontdict={'color':'Darkred', 'size':20});    
        
    if save!=False:
        plt.savefig('C:\\Users\\Egon\\Desktop\\tesi\\immagini tesi\\modelli\\'+title)