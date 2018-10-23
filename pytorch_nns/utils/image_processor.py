import warnings
import numpy as np
import rasterio as rio
from rasterio.windows import Window
#
# CONFIG
#
EPS=1e-12
BANDS_FIRST_AXES=(1,2)
BANDS_LAST_AXES=(0,1)

#
# 256 Stats
#
# S2_MEANS=np.array([
#     496.532625,
#     760.189516,
#     880.292842,
#     2216.19650,
#     772.782279,
#     1290.61584,
#     507.042546])
# S2_STDEVS=np.array([
#     137.294333,
#     112.335618,
#     82.5510594,
#     578.950879,
#     169.387033,
#     386.785660,
#     214.772352])


S2_MEANS=np.array([
    773.9776,
    910.4287,
    998.0203,
    2265.1910,
    1065.8807,
    1960.4547,
    1028.3325,
    0])
S2_STDEVS=np.array([
    209.8368,
    128.6319,
    96.7438,
    462.4491,
    209.3577,
    500.0125,
    360.4749,
    EPS])
S1_MEANS=np.array([65.07169457, 89.35286951])
S1_STDEVS=np.array([31.61495637, 39.82110277])
S2S1_MEANS=np.concatenate([
    S2_MEANS,
    S1_MEANS])
S2S1_STDEVS=np.concatenate([
    S2_STDEVS,
    S1_STDEVS])
CALIBRATION_TARGET_MEANS=[82.41699996948242, 90.44911727905273, 80.29953186035156]
CALIBRATION_TARGET_STDEVS=[45.1631210680867, 39.24848875286493, 36.54758780719321]


#
# HELPERS
# 
def is_bands_first(axes):
    if isinstance(axes,int):
        return axes==0
    else:
        return axes==BANDS_FIRST_AXES


def to_vector(arr):
    return np.array(arr).reshape(-1,1,1)


def image_data(path):
    with rio.open(path,'r') as src:
        profile=src.profile
        image=src.read()
    return image, profile


def image_write(im,path,profile):
    with rio.open(path,'w',**profile) as dst:
        dst.write(im)


""" Crop Image
"""
def crop(image,size,axes=BANDS_FIRST_AXES):
    if is_bands_first(axes):
        return image[:,size:-size,size:-size]
    else:
        return image[size:-size,size:-size]


""" Center Image
"""
def center(image,means=None,to_int=True,axes=BANDS_FIRST_AXES):
    if means is None:
        means=np.mean(image,axis=axes)
    if is_bands_first(axes):
        means=to_vector(means)
    image=(image-means)
    if to_int:
        image=image.round().astype('uint8')
    return image


""" Normalize Image
"""
def normalize(image,means=None,stdevs=None,axes=BANDS_FIRST_AXES):
    if stdevs is None:
        stdevs=np.std(image,axis=axes)   
    image=center(image,means=means,to_int=False,axes=axes)
    if is_bands_first(axes):
        stdevs=to_vector(stdevs)
    return image/stdevs


""" Normalize Difference

    Args:
        b1<int>: the band number for b1
        b2<int>: the band number for b2  


    Returns:
        <arr>: (b1-b2)/(b1+b2)
"""
def normalized_difference(image,band_1,band_2,axes=BANDS_FIRST_AXES):
    if is_bands_first(axes):
        band_1=image[band_1]
        band_2=image[band_2]
    else:
        band_1=image[:,:,band_1]
        band_2=image[:,:,band_2]
    return np.divide(band_1-band_2,band_1+band_2+EPS)



""" Linear Combination

    Args:
        bands <list|int>: list of band indices or band index
        coefs <list|int|None>: list of coefs, a coef for all bands, or None -> 1
        constant <float[0]>: additive constant
    Returns:
        <image>: image_bands dot coefs
"""
def linear_combo(image,bands,coefs=None,constant=None,axes=BANDS_FIRST_AXES):
    if not constant:
        constant=0
    if isinstance(bands,int):
        bands=[bands]
    if not coefs:
        coefs=1
    if isinstance(coefs,int):
        coefs=[coefs]*len(bands)
    if is_bands_first(axes):
        image=coefs[0]*image[bands[0]]
        for c,b in zip(coefs[1:],bands[1:]):
            image+=c*image[b]
    else:
        image=coefs[0]*image[:,:,bands[0]]
        for c,b in zip(coefs[1:],bands[1:]):
            image+=c*image[:,:,b]
    return image+constant


""" Ratio Index
    
    Generalized Index that allows for any linear combination of bands
    in numerator and denominator.

"""
def ratio_index(
        image,
        numerator_bands,
        denominator_bands=None,
        numerator_coefs=None,
        denominator_coefs=None,
        numerator_constant=0,
        denominator_constant=0,
        constant=0):
    if not constant:
        constant=0
    numerator=linear_combo(
        image,
        bands=numerator_bands,
        coefs=numerator_coefs,
        constant=numerator_constant)
    if denominator_bands is None:
        denominator=1
    else:  
        denominator=linear_combo(
            image,
            bands=denominator_bands,
            coefs=denominator_coefs,
            constant=denominator_constant)
    return np.divide(numerator,denominator+EPS)+constant



""" Calibrate Image
"""
def calibrate(image,means=None,stdevs=None,bands=None,axes=BANDS_FIRST_AXES):
    """ WIP:
    
     * RETURNS BANDS LAST IMAGE
     * NEED TO GET TRUE REF COLORS FROM JPEGS
     * USING RGB REFS FROM Kaggle-Planet-Comp

    """
    bands_first=is_bands_first(axes)
    if bands_first:
        nb_bands=image.shape[0]
    else:
        nb_bands=image.shape[-1]
    if nb_bands<3:
        raise ValueError('calibrate requires image with at least 3 bands')
    elif nb_bands>3:
        if bands_first:
            if bands: image=image[bands]
            else: image=image[:3]
        else:
            if bands: image=image[:,:,bands]
            else: image=image[:,:,:3]
    if bands_first:
        image=image.swapaxes(0,1).swapaxes(1,2)
    image=normalize(image,means=means,stdevs=stdevs,axes=BANDS_LAST_AXES)
    image=(image*CALIBRATION_TARGET_STDEVS)+CALIBRATION_TARGET_MEANS
    image=np.clip(image,0,255)
    return image.astype('uint8')


""" Augment Image
"""
def augment(image,k=None,flip=None):
    if k: image=np.rot90(image,k,axes=BANDS_FIRST_AXES)
    if flip: image=np.flip(image,BANDS_FIRST_AXES[0])
    return image


class GTiffLoader(object):


    @staticmethod
    def augmentation(k=None,flip=None):
        if k is None: 
            k=np.random.choice(range(4))
        if flip is None: 
            flip=np.random.choice([True,False])
        return k, flip


    """ 
    """
    def __init__(self,path,cropping=None,bands=None):
        self.path=path
        self.cropping=cropping
        self.bands=bands
        self._set_properities()
        self._set_data()
        



    """ Center Image
    """
    def center(self,means=None,to_int=True):
        self.image=center(self.image,means=means,to_int=to_int)
    
    
    """ Normalize Image
    """
    def normalize(self,means=None,stdevs=None):
        self.image=normalize(self.image,means=means,stdevs=stdevs)


    """ Add Normalized Difference Band
    """
    def normalized_difference(self,band_1,band_2):
        ndiff_band=normalized_difference(self.image,band_1,band_2)
        ndiff_band=np.expand_dims(ndiff_band,axis=0)
        self.image=np.concatenate((self.image,ndiff_band))


    """ Add Ratio Index Band
    """
    def ratio_index(self,
            numerator_bands,
            denominator_bands=None,
            numerator_coefs=None,
            denominator_coefs=None,
            numerator_constant=None,
            denominator_constant=None,
            constant=None):
        ratio_index_band=ratio_index(
            self.image,
            numerator_bands=numerator_bands,
            denominator_bands=denominator_bands,
            numerator_coefs=numerator_coefs,
            denominator_coefs=denominator_coefs,
            numerator_constant=numerator_constant,
            denominator_constant=denominator_constant,
            constant=constant)
        ratio_index_band=np.expand_dims(ratio_index_band,axis=0)
        self.image=np.concatenate((self.image,ratio_index_band))


    """ Calibrate Image
    """
    def calibrate(self,means=None,stdevs=None,bands=None):
        self.image=calibrate(
            self.image,
            means=means,
            stdevs=stdevs,
            bands=bands)


    """ Rotate/Flip Image
    """     
    def augment(self,k=None,flip=None):
        self.k, self.flip=GTiffLoader.augmentation(k,flip)
        self.image=augment(self.image,k,flip)


    """ Map Image Values
    """            
    def value_map(self,value_map,not_value=0):
        value_images=[]
        for k,v in value_map.items():
            value_images.append(np.where(np.isin(self.image,v),k,np.nan))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flat_image=np.nanmax(value_images,axis=0)
        if not_value is None:
            not_value=self.image
        self.image=np.where(np.isnan(flat_image),not_value,flat_image)
        

    """ To Categorical
    """  
    def to_categorical(self,value_list,not_band=False,prepend_not_band=True):
        if self.image.ndim==3:
            self.image=self.image[0]
        im_bands=[(self.image==v).astype(int) for v in value_list]
        if not_band:
            not_image=np.logical_not(np.isin(self.image,value_list))
            if prepend_not_band:
                im_bands=[not_image.astype(int)]+im_bands
            else:
                im_bands=im_bands+[not_image.astype(int)]
        self.image=np.stack(im_bands)

    
    #
    # INTERNAL METHODS
    #
    def _set_properities(self):
        self.k=None
        self.flip=None


    def _set_data(self):
        if isinstance(self.path,str):
            self.image, self.profile=self._image_and_profile(
                self.path,
                self.bands)
        elif isinstance(self.path,list):
            imgs=[]
            if not self.bands:
                self.bands=[None]*len(self.path)
            for path,bnds in zip(self.path,self.bands):
                img,prfl=self._image_and_profile(path,bnds)
                imgs.append(img)
            self.image=np.vstack(imgs)
            self.profile=prfl
        else:
            raise ValueError('GTiffLoader._set_data: path must be <str> or <list<str>>')
        self.profile['count']=self.image.shape[0]
        self.size=self.image.shape[1]



    def _image_and_profile(self,path,bands):
        with rio.open(path) as src:
            profile=src.profile
            image=src.read()
            size=image.shape[1]
            if self.cropping:
                image=crop(image,self.cropping)
                out_size=image.shape[1]
                win=Window(self.cropping,self.cropping,out_size,out_size)
                transform=src.window_transform(win)
                profile=self._crop_profile(profile,out_size,transform)
        if bands:
            image=image[bands]
        return image, profile

    
    def _crop_profile(self,profile,out_size,transform):
        profile=profile.copy()
        profile['width']=out_size
        profile['height']=out_size
        profile['transform']=transform
        if profile['blockxsize']>out_size:
            n=int(np.log(out_size)/np.log(2))
            blocksize=2**n
            profile['blockxsize']=blocksize
            profile['blockysize']=blocksize
        return profile


