# %%
# Import libraries from numpy and scipy
import glob
import os
import tarfile
import re
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp1d, interp2d
from scipy import ndimage
from scipy.optimize import curve_fit
from PIL import Image
import gc
import progressbar
from FftGauss import fft_gauss
import theoryContini as theo
from itertools import chain

# # Define some auxiliary math functions
# 
# * Gaussian profile
# * Double Gaussian
# * Sigmoid function
# * Exponential function
# * Self-normalize function: normalize the input array to 0-1 range
# * fft-gauss: smooth input array by appling a gaussian filter on frequency space
def gauss(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2.*sigma**2))


def dgauss(x, a1, a2, x0, sigma1, sigma2):
    return gauss(x, a1, x0, sigma1) + gauss(x, a2, x0, sigma2)


def sigmoid(x, x0, k, a1=1):
    return a1 / (1 + np.exp(-k*(x-x0)))


def lorentz(x, a, x0, gamma):
    return a*(1/(math.pi*gamma))*((gamma**2)/((x-x0)**2+gamma**2))
   

def pseudovoigt(x, x0, sigma, gamma, nu):
    return (nu*lorentz(x, 1, x0, gamma)+(1-nu)*gauss(x, 1, x0, sigma))


def expfit(z, a, b1, c1, b2, c2):
    return a+b1*np.exp(c1*z)+b2*np.exp(c2*z)


def self_normalize(in_array):
    arr_soft = in_array
    max_soft = np.amax(arr_soft)
    min_soft = np.amin(arr_soft)
    return (in_array - min_soft)/(max_soft - min_soft)

# Function for loading images
def load_imagesMC(path, query):
    '''
        path: folder of files
        query: search string
    '''
    files = glob.glob(path + query)
    files.sort()
    ascs = []
    for i in files:
        reader = np.loadtxt(i)
        ascs.append((np.array(reader).astype("float")))
    images_arr = np.asarray(ascs)
    print("No. of files loaded: " + str(images_arr.shape[0]))
    return images_arr

def load_imagesTIFF(path, query):
    '''
        path: folder of files
        query: search string
    '''
    files = glob.glob(path + query)
    files.sort()
    ascs = []
    ibar = 0
    with progressbar.ProgressBar(max_value=len(files)) as bar:
        for i in files:
            reader = Image.open(i)
            ascs.append((np.array(reader).astype("float")))
            ibar +=1
            bar.update(ibar)
    images_arr = np.asarray(ascs)
    print("No. of files loaded: " + str(images_arr.shape[0]))
    return images_arr

def load_imagesMCZip(compfile, query):
    '''
        path: folder of files
        query: search string
    '''
    #files = zipfile.ZipFile(query, 'r')
    zip = zipfile.ZipFile(compfile)
    files=[]
    for info in zip.infolist():
        if re.match(query, info.filename):
            files.append(zip.extract(info))
    files.sort()
    ascs = []
    for i in files:
        reader = np.loadtxt(i)
        ascs.append((np.array(reader).astype("float")))
    images_arr = np.asarray(ascs)
    print("No. of files loaded: " + str(images_arr.shape[0]))
    return images_arr


# Function for loading images with source positions in filename
def load_imagesTIFF_Zpath(path, query):
    '''
        path: folder of files
        query: search string
    '''
    files = glob.glob(path + query)
    files.sort()
    ascs = []
    sources = []
    ibar = 0
    with progressbar.ProgressBar(max_value=len(files)) as bar:
        for i in files:
            filename = os.path.basename(i)
            listRawProp = re.split("['_','.']+", filename)
            rawPos = int(listRawProp[2])-1
            reader = Image.open(i)
            ascs.append((np.array(reader).astype("uint16")))
            sources.append(rawPos)
            del(listRawProp)
            del(rawPos)
            ibar += 1
            bar.update(ibar) 
    images_arr = np.asarray(ascs)
    print("No. of files loaded: " + str(images_arr.shape[0]))
    del(ascs)
    del(files)
    
    
    gc.collect()
    return images_arr, sources

def load_imagesTIFF_Zpath_tar(file):
    '''
        path: folder of files
        query: search string
    '''
    tar = tarfile.open(file)
    files = tar.getmembers()
    #files.sort()
    ascs = []
    sources = []
    for i in files:
        data = tar.extractfile(i)
        filename = i.name
        print(filename)
        listRawProp = re.split("['_','.']+", filename)
        rawPos = int(listRawProp[2])-1
        reader = Image.open(data)
        ascs.append((np.array(reader).astype("uint16")))
        sources.append(rawPos)
    images_arr = np.asarray(ascs)
    print("No. of files loaded: " + str(images_arr.shape[0]))
    del(ascs)
    del(files)
    del(listRawProp)
    del(rawPos)
    del(tar)
    gc.collect()
    return images_arr, sources

def load_imagesMC_wsp(path, query, fft_smooth=2, posx = 3, posy = 4):
    '''
        path: folder of files
        query: search string
    '''
    files = glob.glob(path + query)
    files.sort()
    ascs = []
    sources = []
    ibar = 0
    with progressbar.ProgressBar(max_value=len(files)) as bar:
        for i in files:
            filename = os.path.basename(i)
            listRawProp = re.split("['_','_']+", filename)
            reader = np.loadtxt(i, dtype="float32")
            # ascs.append(fft_gauss(imresize(np.array(reader).astype("float"),resize,interp="nearest"),fft_smooth)[0])
            ascs.append(fft_gauss(reader,fft_smooth)[0].astype("float16"))
            #ascs.append(reader)
            sources.append([float(listRawProp[posx]),float(listRawProp[posy])])
            del(reader)
            ibar += 1
            bar.update(ibar)                    
    images_arr = np.asarray(ascs)
    del(ascs)
    del(files)
    del(listRawProp)
    gc.collect()
    print("No. of files loaded: " + str(images_arr.shape[0]))
    return images_arr, sources


# ## Define functions for maximum finding in 2d arrays
# * max-pos-fit: fit two 1d gaussians in _x=0_ and _y=0_ and compute the coordinates of the maxima using this gaussians functions
# * max-pos-cm: smooth the image using fft_gauss, discretize it to two values above and below a factor of the maximum, and compute the center of mass. 
# 
# max-pos-cm seems to be better behaived
def max_pos_fit(in_array):
    '''
        Find maximum of the 2d array by fitting two 1d gaussian curves
        in_array: input array
    '''
        
    XFsize = in_array.shape[0]
    YFsize = in_array.shape[1]
    ind_max = np.argmax(in_array)
    ind_max2d = np.unravel_index(ind_max, in_array.shape)

    p0 = [1., 0., 1.]

    xcoeff, var_matrix = curve_fit(gauss, np.arange(XFsize), 
                                   in_array[ind_max2d[0]], p0=p0)
    x_fit = gauss(np.arange(XFsize), *xcoeff)
    x_max = np.argmax(x_fit)

    ycoeff, var_matrix = curve_fit(gauss, np.arange(YFsize), 
                                   in_array[:,ind_max2d[1]], p0=p0)
    y_fit = gauss(np.arange(YFsize), *ycoeff)
    y_max = np.argmax(y_fit)

    return [int(x_max),int(y_max)]


def max_pos_cm(in_array, smooth=40, threshold=0.8):
    '''
        Find maximum of the 2d array by finding the center of 
        mass of a discretized image
        
        in_array: input array
        
        smooth: size of the kernel for the gaussian filter used in 
        smoothing
        
        threshold [0-1]: factor of the maximum to consider for the 
        discretization 
    '''
    if len(in_array) > 0:
        # Smooth image
        im_temp = fft_gauss(in_array, smooth)[0]
    
        # Find the maximim of the smoothed image
        max_im = np.max(im_temp)
    
        # Discretize the image in two levels
        im_temp[im_temp < max_im*threshold] = 0
        im_temp[im_temp > max_im*threshold] = 1
    
        # Find the center of mass
        cm = ndimage.measurements.center_of_mass(im_temp)
        return[int((cm[1])),int((cm[0]))]    
        # return[int(round(cm[1])),int(round(cm[0]))]


# ## Define image manipulation auxiliary functions
# * tomo_crop: Crop the input array using a square window around a given position. Needed to simulate the "output fiber".
# * pad_image: Pad the input array in x and y directions. Needed to build the reconstructed 3D image.
# * change_contrast: Change the contrast (difference between maximum and minimum values) of an image. Used in an attempt to normalize the influence of the depth of the inclusion in the reconstructed image.
def tomo_crop(in_array, offset_x, offset_y, size_crop, center):
    '''
        Crop an array using a square window around a defined position
        
        in_array: input array
        
        offset_x: offset of the center of ROI, x
        
        offset_y: offset of the center of ROI, y
        
        size_crop: lateral size of the square window
        
        center: tuple (x,y) of the center of the WOI
    '''   
    
    crop_im = in_array[center[1] - size_crop + offset_y:center[1] + size_crop + offset_y+1,
                       center[0] - size_crop + offset_x:center[0] + size_crop + offset_x+1]

    return crop_im

def img_crop_cen(in_array, center, xSize, ySize):
    return in_array[center[0]-xSize:center[0]+xSize, center[0]-ySize:center[0]+ySize]


def pad_image(in_array, x_pad, y_pad):
    '''
        Pad an 2D array in x,y keeping original size
        
        in_array: input array
        
        x_pad: amount of padding in x (positive: to the left, 
        negative to the right)
        
        y_pad: amount of padding in y (positive: up, negative: down)
    '''      
    
    x_size = np.shape(in_array)[0]
    y_size = np.shape(in_array)[1]
    
    if x_pad >= 0:
        x_pad_p = x_pad
        x_pad_n = 0
    elif x_pad < 0:
        x_pad_p = 0
        x_pad_n = abs(x_pad)
        
    if y_pad >= 0:
        y_pad_p = y_pad
        y_pad_n = 0
    elif y_pad < 0:
        y_pad_p = 0
        y_pad_n = abs(y_pad)

    padded_array = np.pad(in_array, ((x_pad_p, x_pad_n),
                                     (y_pad_p, y_pad_n)),
                          mode='edge')[0+x_pad_n:x_size+x_pad_n,
                                       0+y_pad_n:y_size+y_pad_n]

    return np.asarray(padded_array)


def change_contrast(in_array, factor):
    '''
        Change the contrast (difference between maximum and minimum values) of an image
        
        in_array: input array
        
        factor: contrast factor, 1 is equal to original
    '''       
    smooth_array = fft_gauss(in_array, 2)[0]
    minarray = np.amin(smooth_array)
    maxarray = np.amax(smooth_array)
    midvalue = np.average([maxarray, minarray])
    return (factor * (in_array-midvalue)+midvalue)

def change_contrastMinMax(in_array, minval, maxval, smooth=False):
    '''
        Change the contrast (difference between maximum and minimum values) of an image
        
        in_array: input array
        
        factor: contrast factor, 1 is equal to original
    '''
    diff = maxval-minval
    if smooth: 
        smooth_array = fft_gauss(in_array, 2)[0]
    else:
        smooth_array = in_array
        
    minarray = np.amin(smooth_array)
    outarray = smooth_array-minarray
    outarray = outarray/np.max(outarray)*diff
    outarray = outarray +minval
    return outarray

# Shift Function
# Prepare interface with MatLab DG code
# TODO: native python version

import os.path
import os
import sys
from PIL import Image

# Tell oct2py where the Octave execuble is in Windows
if sys.platform == 'win32' or  sys.platform == 'win64':
        print("Windows")
        os.environ['OCTAVE_EXECUTABLE'] = "C:/Octave/Octave-5.1.0.0/mingw64/bin/octave-cli.exe"     

# Import oct2py
# Docs: https://pypi.org/project/oct2py/
from oct2py import octave as oc

# Add .m files folder
oc.addpath("2019_Trajectory/subroutines/")

def shiftFuncDG (off, zSlice, d, mua=0.07, mups=10):
        
        # Define geometry
        #zval=np.arange(0,d,0.1)
        #cutvalue=?

        #define optical properties
        n_in=1.33
        #n_out=1.5
        z_e = 2*theo.AAprox(n_in)*theo.Dcomp(mua,mups)
        
        x=oc.Trajektorie_Trans2D_noStruct(d, mua, mups, z_e, off, zSlice)[0][0]

        zShift = off-x
        # return (np.interp(zSlice, zval, zShift))/d # Why /d?
        return zShift
    
# --------------------------------------------------------------------
# Reconstruction functions
# --------------------------------------------------------------------

# #%%
# zRange = np.arange(0,6,0.2)
# shifts = []
# for z in zRange:
#     shifts.append(shiftFuncDG(2, z, 6))
# plt.plot(zRange, np.asarray(shifts))
# #%%

def tomoReconstruction (inputArray, sourcePxPerCm, angleStep, rOffsetTyp, rOffsetRangePer, 
                        slabThickness, mupsAvg, shiftFunc, numZSlices, 
                        sourceShape, cropWindowSize=2, numRSteps=10, recMethod="percentile"):
    '''Perform the actual reconstruction
    
    inputArray: array with the images taken with different source positions
    sourcePxPerCm: scale of the source images, in pixels per centimeter
    angleStep: step in the circular reconstruction of detectors, in degrees
    rOffsetTyp: typical (center) offset in the radial direction
    rOffsetRangePer: percentage of movement between the maximun and minimun offset in the radial direction

    We have three for loops:
    z-guess -> loop through the z space, in order to generate the
    _slices_ of the 3D reconstrution. z-guess goes from 0 to _d_ in _z-size_ steps.
    _d_ is the bulk thickness, _z-size_ is choosen by the user.
    
    off -> loop through different offset sizes. For each offset we calculate 
    the amount of _shift_ required to "undo" the offset focusing
    in the current z slice. This is done using polynomial function 
    fitted to the predicted shift using the theoretical model.
    Also the amount of _attenuation_ is calculated in order to normalize
    the reconstruction of each offset. This att also comes
    from the theoritcal model as a gaussian fit.
    
    angle -> for each offset loop through the circle of equal offset
    where the "detector" can be located. In this loop we took the sums
    of the crops for the current offset and angle, and we construct the
    image with the values of this sums for each
    input image (and, thus, each source position).
     
    Calcule each slice using circular motion

    recMethod = "percentile" #average, median, percentile, or max

    '''

    #Maximum and minumum depths
    z_min = 0.1
    z_max = slabThickness-0.1
    angles = np.arange (0,360,angleStep)

    offsetPx = rOffsetTyp*sourcePxPerCm
    offsetMax = int(offsetPx + offsetPx*rOffsetRangePer/100)
    offsetMin = int(offsetPx - offsetPx*rOffsetRangePer/100)
    offStep = max (1, int((offsetMax - offsetMin)/numRSteps))
    rOffsets = range(offsetMin, offsetMax, offStep)
   
    #Initialize array
    tomo_array = np.zeros((sourceShape[1],sourceShape[0],numZSlices), dtype="float32")

    index_z=0

    #For each slice
    with progressbar.ProgressBar(max_value=numZSlices) as bar:
        for z_guess in np.arange(z_min,z_max,(z_max-z_min)/numZSlices):
            Z_slice=[]
            
            for off in rOffsets:
                
                shift = shiftFunc(off/sourcePxPerCm,z_guess,slabThickness)*sourcePxPerCm
                
                #For each angle
                for angle in angles:
                    
                    #Calculte the shift and offsets in x and y axis
                    x_off = int(off*np.sin(np.radians(angle)))
                    y_off = int(off*np.cos(np.radians(angle)))
                    x_shift = int(shift*np.sin(np.radians(angle)))
                    y_shift = int(shift*np.cos(np.radians(angle)))
                
                    #For each image, take a crop of the image around the 
                    #"detector" position, given by x_off and y_off and 
                    #integrate this crop.
                    img_sums=[]
                    for i in inputArray:
                        crop_sum = np.sum(tomo_crop(i[1],x_off,y_off,cropWindowSize,
                                                i[0]))
                        img_sums.append([i[0][0],i[0][1],crop_sum])
                
                    #Convert the list of sums into a 2D array
                    img_sums_arr= np.asarray(img_sums)
                    img_sums = []
                    
                    #Extract x,y,z from the previous array
                    x_list = img_sums_arr[:,0]
                    y_list = img_sums_arr[:,1]
                    z_list = img_sums_arr[:,2]
                    img_sums_arr = []

                    #Construct the apropiate ranges for x and y
                    min_x=min(x_list)
                    max_x=max(x_list)
                    min_y=min(y_list)
                    max_y=max(y_list)
                    x = np.linspace(min_x, max_x, sourceShape[0])
                    y = np.linspace(min_y, max_y, sourceShape[1])

                    #Construct a meshgrid, needed for the interpolation
                    X,Y = np.meshgrid(x, y)
                    x = []
                    y = []

                    #Interpolate (x,y,z) points over a normal (x,y) grid [X,Y]
                    Z = griddata((x_list, y_list), z_list, (X,Y),  method='linear')
                    x_list = []
                    y_list = []
                    z_list = []
                    
                    #Remove nan values
                    Z[np.isnan(Z)] = 0
                    
                    #Pad the image by a amount given by x_shfit and y_shift
                    Z_pad = pad_image(Z,y_shift,x_shift)
                    Z= []

                    #Append the padded image to a list of images for the given 
                    #slice
                    Z_slice.append(np.flip(change_contrast(Z_pad,(1/att)),axis=0))
                    del(Z_pad)

            #Compute the median of all the images of the current slice
            if recMethod == "median":
                Z_med=np.median(Z_slice, axis=0)
            if recMethod == "percentile":
                Z_med=np.percentile(Z_slice, 95, axis=0)
            if recMethod == "max":
                Z_med=np.ndarray.max(np.asarray(Z_slice), axis=0)
            if recMethod == "average":
                Z_med=np.average(Z_slice, axis=0)


            del(Z_slice)

            #Store the current slice in the 3D matrix
            tomo_array[:,:,index_z]=Z_med
            del(Z_med)
            index_z += 1
            bar.update(index_z)
        

    gc.collect()

    return tomo_array, angles, rOffsets

def tomoMuaReconstruction (inputArray, sourcePxPerCm, angleStep, rOffsetMin, rOffsetMax, 
                           slabThickness, mupsAvg, shiftFunc, numZSlices, 
                           sourceShape, cropWindowSize=2, numRSteps=10, recMethod="average", percVal=5,
                           muaHomo=0.1, mupsHomo=10, mupsInc=10, incSize=0.6):
    '''Perform the actual reconstruction
    
    inputArray: array with the images taken with different source positions
    sourcePxPerCm: scale of the source images, in pixels per centimeter
    angleStep: step in the circular reconstruction of detectors, in degrees
    rOffsetTyp: typical (center) offset in the radial direction
    rOffsetRangePer: percentage of movement between the maximun and minimun offset in the radial direction

    recMethod: percentile, average, median, percentile, min, or max. Def: average

    We have three for loops:
    z-guess -> loop through the z space, in order to generate the
    _slices_ of the 3D reconstrution. z-guess goes from 0 to _d_ in _z-size_ steps.
    _d_ is the bulk thickness, _z-size_ is choosen by the user.
    
    off -> loop through different offset sizes. For each offset we calculate 
    the amount of _shift_ required to "undo" the offset focusing
    in the current z slice. This is done using polynomial function 
    fitted to the predicted shift using the theoretical model.
    Also the amount of _attenuation_ is calculated in order to normalize
    the reconstruction of each offset. This att also comes
    from the theoritcal model as a gaussian fit.
    
    angle -> for each offset loop through the circle of equal offset
    where the "detector" can be located. In this loop we took the sums
    of the crops for the current offset and angle, and we construct the
    image with the values of this sums for each
    input image (and, thus, each source position).
     
    Calcule each slice using circular motion

    '''

    #Maximum and minumum depths
    zMin = 0.1
    zMax = slabThickness-0.1
        
    angles = np.arange (0,360,angleStep)

    offsetMaxPx = int(rOffsetMax*sourcePxPerCm)
    offsetMinPx = int(rOffsetMin*sourcePxPerCm)
    offStepPx = max (1, int((offsetMaxPx - offsetMinPx)/numRSteps))
    rOffsetsPx = range(offsetMinPx, offsetMaxPx, offStepPx)
    if len(rOffsetsPx) != numRSteps: print ("Warning: numRSteps does not match pixel layout, using " + str(len(rOffsetsPx)))
    
    meanIncPath = theo.meanPathLengthT(0,muaHomo,mupsInc,incSize)
   
    #Initialize array
    tomoArray = np.zeros((sourceShape[1],sourceShape[0],numZSlices), dtype="float32")

    indexZ = 0
    indexBar = 0

    #For each slice
    with progressbar.ProgressBar(max_value=numZSlices*len(rOffsetsPx)*len(angles)) as bar:

        for zGuess in np.arange(zMin,zMax,(zMax-zMin)/numZSlices):
            zSlice=[]
            
            for offPx in rOffsetsPx:
                
                shiftPx = shiftFunc(offPx/sourcePxPerCm,zGuess,slabThickness, mua=muaHomo)*sourcePxPerCm/6
                                
                #For each angle
                for angle in angles:
                    
                    bar.update(indexBar)
                    indexBar += 1
                    
                    #Calculate the shift and offsets in x and y axis
                    xOffPx = int(offPx*np.sin(np.radians(angle)))
                    yOffPx = int(offPx*np.cos(np.radians(angle)))
                    xShiftPx = int(shiftPx*np.sin(np.radians(angle)))
                    yShiftPx = int(shiftPx*np.cos(np.radians(angle)))
                
                    #For each image, take a crop of the image around the 
                    #"detector" position, given by x_off and y_off and 
                    #integrate this crop.
                    imgSums=[]
                    for i in inputArray:
                        cropSum = np.sum(tomo_crop(i[1],xOffPx,yOffPx,cropWindowSize, i[0]))
                        imgSums.append([i[0][0],i[0][1],cropSum])
                
                    #Convert the list of sums into a 2D array
                    imgSumsArr= np.asarray(imgSums)
                    imgSums = []
                    
                    #Extract x,y,z from the previous array
                    xList = imgSumsArr[:,0]
                    yList = imgSumsArr[:,1]
                    zList = imgSumsArr[:,2]
                    imgSumsArr = []
                    
                    #Compute the mua of inclusion, assume perturbation model       
                    zMua = muaHomo + np.log(zList/np.average(zList))/(-meanIncPath) 

                    #Construct the apropiate ranges for x and y
                    minX=min(xList)
                    maxX=max(xList)
                    minY=min(yList)
                    maxY=max(yList)
                    x = np.linspace(minX, maxX, sourceShape[0])
                    y = np.linspace(minY, maxY, sourceShape[1])

                    #Construct a meshgrid, needed for the interpolation
                    X,Y = np.meshgrid(x, y)
                    x = []
                    y = []

                    #Interpolate (x,y,z) points over a normal (x,y) grid [X,Y]
                    Z = griddata((xList, yList), zMua, (X,Y),  method='linear')
                    xList = []
                    yList = []
                    zList = []
                    
                    #Remove nan values
                    Z[np.isnan(Z)] = 0
                    
                    #Pad the image by a amount given by x_shfit and y_shift
                    Z_pad = np.abs(pad_image(Z,yShiftPx,xShiftPx))
                    Z= []

                    #Append the padded image to a list of images for the given 
                    #slice
                    zSlice.append(np.flip((Z_pad),axis=0))
                    
                    del(Z_pad)

            #Compute the median of all the images of the current slice
            if recMethod == "median":
                zMed=np.median(zSlice, axis=0)
            if recMethod == "percentileMin":
                zMed=np.percentile(zSlice, percVal, axis=0)
            if recMethod == "percentileMax":
                zMed=np.percentile(zSlice, 100-percVal, axis=0)
            if recMethod == "max":
                zMed=np.ndarray.max(np.asarray(zSlice), axis=0)
            if recMethod == "min":
                zMed=np.ndarray.min(np.asarray(zSlice), axis=0)
            if recMethod == "average":
                zMed=np.average(zSlice, axis=0)


            del(zSlice)

            #Store the current slice in the 3D matrix
            tomoArray[:,:,indexZ]=zMed
            del(zMed)
            indexZ += 1
        

    gc.collect()

    return tomoArray, angles, rOffsetsPx



def tomoMuaReconstructionV2 (inputArray, sourcePxPerCm, rMin, rMax,
                             slabThickness, mupsAvg, shiftFunc, numZSlices, 
                             sourceShape, cropWindowSize=2, numRSteps=20, recMethod="average", 
                             percVal=5, muaHomo=0.1, mupsHomo=10, mupsInc=10, incSize=0.6, 
                             inputBinning=1, rBinning=1):
    '''Perform the actual reconstruction
    
    inputArray: array with the images taken with different source positions
    sourcePxPerCm: scale of the source images, in pixels per centimeter
    angleStep: step in the circular reconstruction of detectors, in degrees
    rOffsetTyp: typical (center) offset in the radial direction
    rOffsetRangePer: percentage of movement between the maximun and minimun offset in the radial direction

    recMethod: percentile, average, median, percentile, min, or max. Def: average

    We have three for loops:
    z-guess -> loop through the z space, in order to generate the
    _slices_ of the 3D reconstrution. z-guess goes from 0 to _d_ in _z-size_ steps.
    _d_ is the bulk thickness, _z-size_ is choosen by the user.
    
    off -> loop through different offset sizes. For each offset we calculate 
    the amount of _shift_ required to "undo" the offset focusing
    in the current z slice. This is done using polynomial function 
    fitted to the predicted shift using the theoretical model.
    Also the amount of _attenuation_ is calculated in order to normalize
    the reconstruction of each offset. This att also comes
    from the theoritcal model as a gaussian fit.
    
    angle -> for each offset loop through the circle of equal offset
    where the "detector" can be located. In this loop we took the sums
    of the crops for the current offset and angle, and we construct the
    image with the values of this sums for each
    input image (and, thus, each source position).
     
    Calcule each slice using circular motion

    '''

    #Maximum and minumum depths
    zMin = 0.1
    zMax = slabThickness-0.1
    zRange = np.arange(zMin,zMax,(zMax-zMin)/numZSlices)
    
    
    offsetMaxPx = int(rMax*sourcePxPerCm)
    offsetMinPx = int(rMin*sourcePxPerCm)
    boxMaxPx = 2*offsetMaxPx
    boxMinPx = 0#int(math.sqrt(2*offsetMinPx**2)/2)-1
    xyRangePx = range(boxMinPx, boxMaxPx, rBinning)
    
    print(rMin, rMax, offsetMaxPx, offsetMinPx, xyRangePx)
    
    meanIncPath = theo.meanPathLengthT(0,muaHomo,mupsInc,incSize)
   
    #Initialize array
    tomoArray = np.zeros((sourceShape[1],sourceShape[0],numZSlices), dtype="float32")

    indexZ = 0

    #For each slice
    with progressbar.ProgressBar(max_value=numZSlices) as bar:
        for zGuess in zRange:
            zSlice=[]
          
            # Every pixel in boxed region, centered at detector, 
            # of boxMin < x,y < boxMax is considered as a detector
            for xOffPx in xyRangePx:
                for yOffPx in xyRangePx:
                
                    # Distance to the center 
                    xOffCenPx = xOffPx - offsetMaxPx   
                    yOffCenPx = yOffPx - offsetMaxPx
                    xyOffPx = math.sqrt(xOffCenPx**2 + yOffCenPx**2)                              
                    
                    # If distance to cented is less than offsetMax and more than offsetMin
                    if (xyOffPx <= offsetMaxPx and xyOffPx >= offsetMinPx):
                       
                        angleOff = math.atan(yOffCenPx/xOffCenPx)
                        
                        modShiftPx = shiftFunc(xyOffPx/sourcePxPerCm,zGuess,slabThickness, mua=muaHomo)*sourcePxPerCm/6
                        
                        xShiftPx = int(modShiftPx*math.sin(angleOff))
                        yShiftPx = int(modShiftPx*math.cos(angleOff))                        
                                         
                        imgSums=[]
                        for i in inputArray[::inputBinning]:
                            x = int(i[0][1]+xOffCenPx)
                            y = int(i[0][0]+yOffCenPx)
                            imgSums.append([i[0][0],i[0][1],i[1][x,y]])
                    
                        #Convert the list of sums into a 2D array
                        imgSumsArr= np.asarray(imgSums)
                        imgSums = []
                        
                        #Extract x,y,z from the previous array
                        xList = imgSumsArr[:,0]
                        yList = imgSumsArr[:,1]
                        zList = imgSumsArr[:,2]
                        imgSumsArr = []
                        
                        #Compute the mua of inclusion, assume perturbation model        
                        zMua = muaHomo + np.log(zList/np.average(zList))/(-meanIncPath) 
                                                        
                        #Construct the apropiate ranges for x and y
                        minX=min(xList)
                        maxX=max(xList)
                        minY=min(yList)
                        maxY=max(yList)
                        x = np.linspace(minX, maxX, sourceShape[0])
                        y = np.linspace(minY, maxY, sourceShape[1])

                        #Construct a meshgrid, needed for the interpolation
                        X,Y = np.meshgrid(x, y)
                        x = []
                        y = []

                        #Interpolate (x,y,z) points over a normal (x,y) grid [X,Y]
                        Z = griddata((xList, yList), zMua, (X,Y),  method='linear')
                        xList = []
                        yList = []
                        zList = []
                        
                        #Remove nan values
                        Z[np.isnan(Z)] = 0
                        
                        #Pad the image by a amount given by x_shfit and y_shift
                        zPad = pad_image(Z,yShiftPx,xShiftPx)
                        Z= []

                        #Append the padded image to a list of images for the given 
                        #slice
                        zSlice.append(np.flip((zPad),axis=0))
                        del(zPad)
                    

            #Compute the median of all the images of the current slice
            if recMethod == "median":
                zMed=np.median(zSlice, axis=0)
            if recMethod == "percentile":
                zMed=np.percentile(zSlice, percVal, axis=0)
            if recMethod == "max":
                zMed=np.ndarray.max(np.asarray(zSlice), axis=0)
            if recMethod == "min":
                zMed=np.ndarray.min(np.asarray(zSlice), axis=0)
            if recMethod == "average":
                zMed=np.average(zSlice, axis=0)


            del(zSlice)
            

            #Store the current slice in the 3D matrix
            tomoArray[:,:,indexZ]=zMed
            indexZ += 1
            bar.update(indexZ)
            del(zMed)
            
            
        

    gc.collect()

    return tomoArray


# %%
