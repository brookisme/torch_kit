#
# BAND INDICES
#
# For use with:
#   - GTiffLoader.ratio_index
#   - GTiffLoader.normalized_difference
#
ratio={
    'greeness':{ 
        "numerator_bands":1,
        "denominator_bands":0,
    },
   'chlogreen':{
        "numerator_bands":3,
        "denominator_bands":[1,4],
    },
    'gcvi':{
        "numerator_bands":3,
        "denominator_bands":1,
        "numerator_constant":1
    },
    'evi_modis':{
        "numerator_bands":[3,0],
        "numerator_coefs":[2.5,-2.5],
        "denominator_bands":[3,0,2],
        "denominator_coefs":[1,6,7.5],
        "denominator_constant":1
    },
    'evi_s2':{
        "numerator_bands":[3,0],
        "numerator_coefs":[2.5,-2.5],
        "denominator_bands":[3,0,2],
        "denominator_coefs":[1,6,7.5],
        "denominator_constant":10000
    }
}
normalized_difference={
    'ndvi':(3,0),
    'ndwi':(1,3),
    'ndwi_leaves':(3,5)
}
