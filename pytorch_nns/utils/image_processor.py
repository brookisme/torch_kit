import warnings
import numpy as np
import rasterio as rio
#
# CONFIG
#
EPS=1e-12
AXIS=(1,2)
S2_MEANS=np.array([
    496.532625,
    760.189516,
    880.292842,
    2216.19650,
    772.782279,
    1290.61584,
    507.042546])
S2_STDEVS=np.array([
    137.294333,
    112.335618,
    82.5510594,
    578.950879,
    169.387033,
    386.785660,
    214.772352])
CALIBRATION_TARGET_MEANS=[82.41699996948242, 90.44911727905273, 80.29953186035156]
CALIBRATION_TARGET_STDEVS=[45.1631210680867, 39.24848875286493, 36.54758780719321]


#
# HELPERS
# 
def is_bands_first(axis):
    return axis==(1,2)


def to_vector(arr):
    return np.array(arr).reshape(-1,1,1)


def image_data(path):
    with rio.open(path) as src:
        profile=src.profile
        image=src.read()
    return image, profile


""" Crop Image
"""
def crop(image,size,axis=AXIS):
    if is_bands_first(axis):
        return image[:,size:-size,size:-size]
    else:
        return image[size:-size,size:-size]


""" Center Image
"""
def center(image,means=None,to_int=True,axis=AXIS):
    if means is None:
        means=np.mean(image,axis=axis)
    if is_bands_first(axis):
        means=to_vector(means)
    image=(image-means)
    if to_int:
        image=image.round().astype('uint8')
    return image


""" Normalize Image
"""
def normalize(image,means=None,stdevs=None,axis=AXIS):
    if stdevs is None:
        stdevs=np.std(image,axis=axis)   
    image=center(image,means=means,to_int=False,axis=axis)
    if is_bands_first(axis):
        stdevs=to_vector(stdevs)
    return image/stdevs


""" Normalize Difference

    Args:
        b1<int>: the band number for b1
        b2<int>: the band number for b2  


    Returns:
        <arr>: (b1-b2)/(b1+b2)
"""
def normalized_difference(image,band_1,band_2,axis=AXIS):
    if is_bands_first(axis):
        band_1=image[band_1]
        band_2=image[band_2]
    else:
        band_1=image[:,:,band_1]
        band_2=image[:,:,band_2]
    return np.divide(band_1-band_2,band_1+band_2+EPS)


""" Calibrate Image
"""
def calibrate(image,means=None,stdevs=None,bands=None,band_axis=0):
    """ WIP:
    
     * RETURNS BANDS LAST IMAGE
     * NEED TO GET TRUE REF COLORS FROM JPEGS
     * USING RGB REFS FROM Kaggle-Planet-Comp

    """
    nb_bands=image.shape[band_axis]
    if nb_bands<3:
        raise ValueError('calibrate requires image with at least 3 bands')
    elif nb_bands>3:
        if band_axis is 0:
            if bands: image=image[bands]
            else: image=image[:3]
        else:
            if bands: image=image[:,:,bands]
            else: image=image[:,:,:3]
    if band_axis!=-1:
        image=np.swapaxes(image,band_axis,-1)
    image=normalize(image,means=means,stdevs=stdevs)
    image=(image*CALIBRATION_TARGET_STDEVS)+CALIBRATION_TARGET_MEANS
    image=np.clip(image,0,255)
    return image.astype('uint8')


""" Augment Image
"""
def augment(image,k=None,flip=None):
    if k: image=np.rot90(image,k,axes=AXIS)
    if flip: image=np.flip(image,AXIS[0])
    return image


class GTiffLoader(object):
    """
    """
    def __init__(self,path,cropping=None,bands=None):
        self.path=path
        self.cropping=cropping
        self.bands=bands
        self._set_properities()
        self._set_data()
        if bands:
            self.image=self.image[bands]



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


    """ Rotate/Flip Image
    """     
    def augment(self,k=None,flip=None):
        if k is None: 
            k=np.random.choice(range(4))
        if flip is None: 
            flip=np.random.choice([True,False])
        self.k=k
        self.flip=flip
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
        with rio.open(self.path) as src:
            self.profile=src.profile
            self.image=src.read()
            self.size=self.image.shape[1]
            if self.cropping:
                self.image=crop(self.image,self.cropping)
                self.profile=self._crop_profile(src)

    
    def _crop_profile(self,src):
        out_size=self.size-2*self.cropping
        win=((self.cropping,self.cropping),(out_size,out_size))
        profile=self.profile.copy()
        profile.pop('transform',None)
        profile['width']=out_size
        profile['height']=out_size
        profile['affine']=src.window_transform(win)
        return profile


