

from matplotlib import pyplot as plt
from matplotlib import gridspec, patches
import numpy as np
import pickle
import scipy
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.measure import block_reduce
from skimage.measure import label
from skimage.measure import regionprops
from skimage import exposure, morphology
from time import time
import glob
import os.path
import re


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


# creates output folder, gets image names, finds image size
def preamble():
    global run
    global fileNames
    global fileLoc
    global bacteria_type
    global usrname
    global xpix_length
    global ypix_length
    global output_file
    global cubes
    global ROI_locs
    folder_location = input('copy paste (CTRL+SHIFT+v) the file location of your first image please:  ')
    print()
    befA_conc = input('What is the BefA concentration')
    print()
    scan_number = input('What scan is this?')
    print()
    region = input('Which region is this?')
    print()
    fileLoc = folder_location
    output_loc = folder_location + '/labels/'
    if not os.path.exists(output_loc):
        os.mkdir(output_loc)
    output_file = output_loc + befA_conc + '_scan_' + scan_number + '_region_' + region
    run = 1
    if os.path.isfile(output_file):
        print()
        cubes = textLoader()[0]
        ROI_locs = textLoader()[1]
        run = 0
    fileNames = glob.glob(fileLoc + '/*.tif')
    fileNames.extend(glob.glob(fileLoc + '/*.png'))
    sort_nicely(fileNames)
    pix_dimage = plt.imread(fileNames[0])
    ypix_length = len(pix_dimage[0])
    xpix_length = len(pix_dimage)


# determines distance between two objects
def dist(x1, y1, input_list):
    x2 = input_list[0]
    y2 = input_list[1]
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


# Loop through frames and find potential vesicles
def blob_detection(start, stop, scale):
    global blobs
    global start_time
    global plots
    plots = []
    start_time = time()
    blobs = []
    print('starting loop')
    t_read = 0
    t_reduce = 0
    t_resize = 0
    t_blob = 0
    t_append = 0

    for name in fileNames[start:stop]:
        t0 = time()
        image = plt.imread(name)
        image = rgb2gray(image)
        t1 = time()
        t_read += t1-t0
        image = block_reduce(image, block_size=(scale, scale), func=np.mean)
        plots.append(image.tolist())
        t2 = time()
        t_reduce += t2-t1
        image = (image - np.min(image))/np.max(image)
        t3 = time()
        t_resize += t3-t2

        # Find edges in images, remove small objects, label them, and extract properties
        edges = canny(image, sigma=1, low_threshold=0.1, high_threshold=0.2, use_quantiles=False)
        edges = morphology.binary_dilation(edges)
        edges = scipy.morphology.binary_fill_holes(edges)
        edges = morphology.remove_small_objects(edges, 50)
        labels = label(edges, connectivity=2)
        props = regionprops(labels, image)

        # Save images with their edge counterparts. Used for troubleshooting could delete
        # fig = plt.figure()
        # ax1 = plt.subplot(121)
        # ax1 = plt.imshow(image)
        # ax2 = plt.subplot(122)
        # ax2 = plt.imshow(edges)
        # fig.savefig('/media/chiron/Alderaan/BefA_data/BefA_part6_2_12_20_BefAFLlow/fish1/Scans/scan_2/region_1/568nm/test/' + name[-7:])
        # plt.close()

        # Reformat information from regionprops to tempblobs [xcenter, ycenter, 0, 0, 0, rough radius]
        tempblobs = np.zeros(shape=(len(props), 6))
        for n in range(len(props)):
            tempblobs[n][0:2] = props[n].centroid
            # length of the major axis of the ellipse that has the same normalized second central moments as the region
            tempblobs[n][-1] = props[n].major_axis_length
        tempblobs = tempblobs.tolist()
        t4 = time()
        t_blob += t4-t3
        # If no vesicles found in frame, add this placeholder.
        # Gets removed later when trimming, but could find a less janky solution later
        if not tempblobs:
            blobs.append([[1, 1, 0, 0, 0, 1]])
        else:
            blobs.append(tempblobs)
        t5 = time()
        t_append += t5-t4
        print(name)
    print(blobs)
    print('t_read = ' + str(round(t_read, 1)))
    print('t_reduce = ' + str(round(t_reduce, 1)))
    print('t_resize = ' + str(round(t_resize, 1)))
    print('t_blob = ' + str(round(t_blob, 1)))
    print('t_append = ' + str(round(t_append, 1)))


# Remove blobs that are not above a threshold value
def trim_segmented(blobs, width=30, thresh2=0.7):
    global trim_time
    plots1 = segmentation_mask(plots, width, thresh2)
    trim_time = time()
    print('done building the mask')
    for z in range(len(blobs)):
        rem = []
        for blob in blobs[z]:
            if plots1[z][int(blob[0])][int(blob[1])] is False and blob != []:
                rem.append(blob)
        for item in rem:
            blobs[z].remove(item)
    return blobs


# Create a threshold for each image in a stack
def segmentation_mask(plots1, width, thresh2):
    plots_out = [[] for el in range(len(plots1))]
    plots2 = [[] for el in range(len(plots1))]
    print('building mask...')
    for i in range(len(plots1)):
        image = plots1[i]
        image = (image - np.min(image))/np.max(image)
        plots2[i] = exposure.equalize_hist(np.array(image))
    # Find the average value across 30 images for each image
    for i in range(len(plots2)):
        if i < int(width / 2):
            image = np.mean(plots2[0: width], axis=0)
        elif i > int(len(plots2) - width / 2):
            image = np.mean(plots2[-width: -1], axis=0)
        else:
            image = np.mean(plots2[i-int(width / 2):i + int(width / 2)], axis=0)
        # Use this average to create a threshold
        binary = image > thresh2
        plots_out[i] = binary
    return plots_out


# Combine blobs that are in consecutive frames that are a distance adjSize from one another
def trim_consecutively(blobs, adjSize=2):
    # I believe z is the frame, n is the blob
    # Changes blobs so that blobs present in adjacent frames are counted as one - changes format of blobs:
    # blobs input: [[[x, y, 0, 0, 0, rough radius]...]]
    # blobs output: [[first x, first y, # of frames present in, last  x, last y, rough radius]...]
    for z in range(len(blobs)):
        for n in range(len(blobs[z])):
            if blobs[z][n][5] == 0:
                break
            else:
                blobs[z][n][2] = 1
                contains = 'True'
                zz = z + 1
                test_location = blobs[z][n][0:2]
                while contains == 'True' and zz < len(blobs):
                    if not blobs[zz]:  # check for empty zz
                        break
                    for blob in blobs[zz]:
                        if dist(blob[0], blob[1], test_location) < adjSize:
                            blobs[z][n][2] += 1
                            test_location = blob[0:2]
                            # x-end
                            blobs[z][n][3] = test_location[0]
                            # y-end
                            blobs[z][n][4] = test_location[1]
                            # keep larger rough radius
                            blobs[z][n][5] = max(blobs[z][n][5], blob[5])

                            blobs[zz].remove(blob)
                            zz += 1
                            contains = 'True'
                            break
                        else:
                            contains = 'False'
                # z_stretch = dist(blobx, bloby, firstlocation)
                # blobs[z][n].append(z_stretch)
    return blobs


# trim when blob only in one or two planes
def trim_toofewtoomany(blobs, tooFew=2):
    edgeTrim = 10
    for z in range(len(blobs)):
        rem = []    # note, removing while looping skips every other entry to be removed
        for blob in blobs[z]:
            if blob[2] < tooFew:
                rem.append(blob)
            # the following makes sure blobs aren't on x-y edge of image
            elif blob[0] < edgeTrim or blob[1] < edgeTrim:
                rem.append(blob)
            elif blob[0] > xpix_length - edgeTrim:
                rem.append(blob)
            elif blob[1] > ypix_length - edgeTrim:
                rem.append(blob)
        for item in rem:
            blobs[z].remove(item)
    return blobs


# extracts the cube of the images around where an object is
def cube_extractor():  # Maybe want sliding input_image?
    z = 0
    cubes = [[] for el in ROI_locs]
    for name in fileNames[start:stop]:
        z += 1
        image = plt.imread(name)  # CHANGE TO EXTRACT FROM PLOTS
        for el in range(len(ROI_locs)):
            if ROI_locs[el][2] > len(blobs) - int(zLength / 2) and z > len(blobs) - zLength:
                xstart = int(ROI_locs[el][0] - cubeLength / 2)
                ystart = int(ROI_locs[el][1] - cubeLength / 2)
                subimage = image[xstart:xstart + cubeLength, ystart:ystart + cubeLength].tolist()
                cubes[el].append(subimage)
            elif ROI_locs[el][2] > z + int(zLength / 2):
                break
            elif ROI_locs[el][2] <= int(zLength / 2) and z <= zLength:
                xstart = int(ROI_locs[el][0] - cubeLength / 2)
                ystart = int(ROI_locs[el][1] - cubeLength / 2)
                subimage = image[xstart:xstart + cubeLength, ystart:ystart + cubeLength].tolist()
                cubes[el].append(subimage)
            elif ROI_locs[el][2] > z - int(zLength / 2):
                xstart = int(ROI_locs[el][0] - cubeLength / 2)
                ystart = int(ROI_locs[el][1] - cubeLength / 2)
                subimage = image[xstart:xstart + cubeLength, ystart:ystart + cubeLength].tolist()
                cubes[el].append(subimage)
    print('total time = ' + str(round(time() - start_time, 1)))
    return cubes


# Save ROI_locs and cubes
def textSaver(blibs):
    global cubes2
    cubes2 = [[] for element in cubes]
    print('saving...')
    for el in range(len(blibs)):
        cubes2[el] = [cubes[el], blibs[el][4], blibs[el][0:4]]
    pickle.dump(cubes2, open(output_file, 'wb'))

    print('done saving truth table')


# Load saved information
def textLoader():
    loaded = pickle.load(open(output_file, 'rb'))
    cubes1 = []
    blibs1 = []
    for el in loaded:
        # el[0] is the 10x60x60 image of the blob
        cubes1.append(el[0])
        # el[1] is label, el[2:5][0][0:3] is [x, y, z, radius]
        blibs1.append([el[2:5][0][0], el[2:5][0][1], el[2:5][0][2], el[2:5][0][3], el[1]])
    return [cubes1, blibs1]


# Sets up the control of the GUI for z scan and zooming
def key_z_plots(e):
    global curr_pos
    global background_color
    if e.key == "right":
        curr_pos = curr_pos + 1
    elif e.key == "left":
        curr_pos = curr_pos - 1
    else:
        return
    if zoom == 'on':
        xbegin = max([int(ROI_locs[blobNum][0]) - zoom_width, 0])
        ybegin = max([int(ROI_locs[blobNum][1]) - zoom_width, 0])
        xend = min([int(ROI_locs[blobNum][0]) + zoom_width, xpix_length])
        yend = min([int(ROI_locs[blobNum][1]) + zoom_width, ypix_length])
    elif zoom == 'off':
        xbegin = 0
        ybegin = 0
        xend = -1
        yend = -1
    curr_pos = curr_pos % len(plots)
    plt.cla()
    image = plt.imread(fileNames[curr_pos])
    image = (image - np.min(image)) / np.max(image)
    plt.imshow(image[xbegin:xend, ybegin:yend], cmap=cmaps[color_int])
    plt.gcf().gca().add_artist(r)
    plt.title('z location is: ' + str(curr_pos) + '        ' + 'z center is: ' + str(ROI_locs[blobNum][2]))
    plt.draw()
    if ROI_locs[blobNum][2] < int(zLength / 2):
        if np.abs(curr_pos) > zLength:
            background_color = 'red'
        else:
            background_color = 'gray'
    else:
        if np.abs(curr_pos - ROI_locs[blobNum][2]) > int(zLength / 2):
            background_color = 'red'
        else:
            background_color = 'gray'
    fig.patch.set_facecolor(background_color)


# Sets up GUI control for labelling,
def key_blobs(f):
    global blobNum
    global zoom
    global xbegin
    global xend
    global ybegin
    global yend
    global curr_pos
    global background_color
    if f.key == "up":
        blobNum += 1
    elif f.key == "down":
        blobNum -= 1
    elif f.key == 'shift':
        blobNum = blobNum + 100
    elif f.key == 'control':
        blobNum = blobNum - 100
    elif f.key == '/':
        blobNum += 1
        while ROI_locs[blobNum][-1] != '?' and blobNum < len(ROI_locs) - 1:
            blobNum += 1
    else:
        return
    zoom = 'off'
    blobNum = blobNum % len(ROI_locs)
    plt.cla()
    fig.suptitle('blob number ' + str(blobNum + 1), fontsize=20)
    background_color = 'gray'
    fig.patch.set_facecolor(background_color)
    curr_pos = ROI_locs[blobNum][2]
    plotInit(blobNum)


def key_zoom(h):
    global zoom
    global xbegin
    global xend
    global ybegin
    global yend
    if h.key == 'z':
        if zoom == 'off':
            zoom = 'on'
        elif zoom == 'on':
            zoom = 'off'
        plt.cla()
        fig.suptitle('blob number ' + str(blobNum + 1), fontsize=20)
        fig.patch.set_facecolor(background_color)
        plotInit(blobNum)
    else:
            return


def key_tagging(g):
    global blobNum
    global curr_pos
    global fig
    global color_int
    global gs
    global full_res
    global zoom
    global background_color
    # Label the object and automatically go to the next object
    if g.key == 'c' or g.key == 'n' or g.key == 'v' or g.key == 'm' or g.key == '2':
        ROI_locs[blobNum][-1] = g.key
        blobNum += 1
        blobNum = blobNum % len(ROI_locs)
        zoom = 'off'
        plt.cla()
        fig.suptitle('blob number ' + str(blobNum + 1), fontsize=20)
        background_color = 'gray'
        fig.patch.set_facecolor(background_color)
        curr_pos = ROI_locs[blobNum][2]
        plotInit(blobNum)
    # Skip to the next unlabelled object
    elif g.key == 't':
        startpoint = blobNum
        while True:
            if ROI_locs[blobNum][-1] == '?':
                break
            blobNum += 1
            blobNum = blobNum % len(ROI_locs)
            if blobNum == startpoint:
                break
            zoom = 'off'
            plt.cla()
            fig.suptitle('blob number ' + str(blobNum + 1), fontsize=20)
            background_color = 'gray'
            fig.patch.set_facecolor(background_color)
            curr_pos = ROI_locs[blobNum][2]
            plotInit(blobNum)
    # Change the color scheme
    elif g.key == '.':
        color_int += 1
        color_int = color_int % len(cmaps)
        plotInit(blobNum)
    elif g.key == ',':
        color_int += -1
        color_int = color_int % len(cmaps)
        plotInit(blobNum)
    # Save the labels
    elif g.key == 'enter':
        plt.close()
        textSaver(ROI_locs)
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle('blob number ' + str(blobNum + 1), fontsize=20)
        fig.patch.set_facecolor(background_color)
        plotInit(blobNum)
        plt.show()
    # Close the GUI, save the labels, and print the counts
    elif g.key == 'escape':
        plt.close()
        textSaver(ROI_locs)
        # Print number of each type:
        unique, counts = np.unique(ROI_locs, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        print('Number of features:')
        if '?' in counts_dict.keys():
            print('Unlabeled: ' + str(counts_dict['?']))
        else:
            print('Unlabeled: None')
        if 'n' in counts_dict.keys():
            print('Nothing: ' + str(counts_dict['n']))
        else:
            print('Nothing: None')
        if 'v' in counts_dict.keys():
            print('Vesicles: ' + str(counts_dict['v']))
        else:
            print('Vesicles: None')
        if 'c' in counts_dict.keys():
            print('Clusters: ' + str(counts_dict['c']))
        else:
            print('Clusters: None')
        # Add print of average vesicle radius, % clusters
        if 'c' in counts_dict.keys() and 'v' in counts_dict.keys():
            print('Total features: ' + str(counts_dict['c'] + counts_dict['v']))
            print('% clusters: ' + str(counts_dict['c']/(counts_dict['c'] + counts_dict['v'])*100) + '%')
        ROI_array = np.asarray(ROI_locs)
        vesicles = ROI_array[ROI_array[:, -1] == 'v']
        avg_radius = np.mean(vesicles[:, 3].astype(np.float))
        print('Average radius of vesicles: ' + str(avg_radius) + ' pixels')
    else:
        return


def cubePlots(blobNum):
    plt.subplot(gs[0, -1])
    image = np.amax(cubes[blobNum], axis=0)
    plt.imshow(image, cmap=cmaps[color_int])
    plt.subplot(gs[1, -1])
    image = np.amax(cubes[blobNum], axis=1)
    plt.imshow(image, cmap=cmaps[color_int])
    plt.subplot(gs[2, -1])
    image = np.amax(cubes[blobNum], axis=2)
    plt.imshow(image, cmap=cmaps[color_int])


def plotInit(blobNum):
    global gs
    global r
    global curr_pos
    global iterList
    global cmaps
    global zoom
    if zoom == 'on':
        xbegin = int(max([ROI_locs[blobNum][0] - zoom_width, 0]))
        ybegin = int(max([ROI_locs[blobNum][1] - zoom_width, 0]))
        xend = int(min([ROI_locs[blobNum][0] + zoom_width, xpix_length]))
        yend = int(min([ROI_locs[blobNum][1] + zoom_width, ypix_length]))
    elif zoom == 'off':
        xbegin = 0
        ybegin = 0
        xend = -1
        yend = -1
    cmaps = ['viridis', 'bone', 'inferno', 'BrBG', 'gist_rainbow', 'gnuplot', 'ocean', 'Paired', 'Set1']
    plt.cla()
    cubePlots(blobNum)
    plt.subplot(gs[-1, -1])
    plt.cla()
    plt.axis('off')
    plt.text(0.25, 0.5, 'label:  ' + str(ROI_locs[blobNum][-1]), fontsize=40)
    plt.subplot(gs[:, 0:4])
    fig.canvas.mpl_connect('key_press_event', key_z_plots)
    fig.canvas.mpl_connect('key_press_event', key_blobs)
    fig.canvas.mpl_connect('key_press_event', key_tagging)
    fig.canvas.mpl_connect('key_press_event', key_zoom)
    image = plt.imread(fileNames[curr_pos])
    image = (image - np.min(image)) / np.max(image)
    plt.imshow(image[xbegin:xend, ybegin:yend], cmap=cmaps[color_int])
    y, x = [ROI_locs[blobNum][i] - [xbegin, ybegin][i] for i in range(2)]
    sidelength = 2*2*ROI_locs[blobNum][3]
    r = patches.Rectangle((x - sidelength/2, y - sidelength/2), sidelength, sidelength, color='red', linewidth=1, fill=False)
    plt.title('z location is: ' + str(curr_pos) + '        ' + 'z center is: ' + str(ROI_locs[blobNum][2]))
    plt.gcf().gca().add_artist(r)
    plt.draw()


########################################################################################################################
#                                               SET UP
preamble()
########################################################################################################################
#                                               CREATE 3D BLOBS LIST
#                                (by looping through and blob-detecting each image)

start = 0
stop = -1
scale = 4
cubeLength = 150
zLength = 20
zoom_width = 200
if run == 1:

    blob_detection(start, stop, scale)

    ####################################################################################################################
    #                                     TRIMMING LIST OF BLOBS                                                       #

    blobs = trim_segmented(blobs)
    blobs = trim_consecutively(blobs)
    blobs = trim_toofewtoomany(blobs)
    print('Total time to trim blobs = ' + str(round(time() - trim_time, 1)))

    # blibs is one-d list of (x,y,z, bacType) for detected blobs
    # ROI_locs = [y position, x position, z position, radius, label]
    ROI_locs = [[blobs[i][n][0] * scale + (blobs[i][n][3] - blobs[i][n][0]) / 2 * scale,
                 blobs[i][n][1] * scale + (blobs[i][n][4] - blobs[i][n][1]) / 2 * scale,
                 int(i + blobs[i][n][2] / 2),
                 blobs[i][n][5]] for i in range(len(blobs)) for n in range(len(blobs[i]))]
    # blibs = [[blobs[i][n][0]*scale, blobs[i][n][1]*scale, int(i + blobs[i][n][2]/2)] for i in range(len(blobs))
    #          for n in range(len(blobs[i]))]
    ROI_locs = sorted(ROI_locs, key=lambda x: x[2])
    [blip.append('?') for blip in ROI_locs]

    ####################################################################################################################
    #                                           CUBE EXTRACTOR                                                         #
    #                          ( extract a input_image around each blob for classification )                           #
    #                          ( cubes is indexed by blob, z, x,y )                                                    #
    cubes = cube_extractor()

else:
    plots = []
    for name in fileNames[start:stop]:
        image = plt.imread(name)
        image = block_reduce(image, block_size=(scale, scale), func=np.mean)
        print(name)
        plots.append(image.tolist())
########################################################################################################################
#                                           IMAGE BUILDER                                                              #


print(str(len(cubes)) + ' detected vesicles')
# for blip in blibs:
#     if blip[-1] == 'b':
#         blibs.remove(blip)
#     if blip[-1] == '?':
#         blip[-1] = 'n'

blobNum = 0
while ROI_locs[blobNum][-1] != '?' and blobNum < len(ROI_locs)-1:
    blobNum += 1
color_int = 5
xbegin = 0
ybegin = 0
xend = -1
yend = -1
curr_pos = ROI_locs[blobNum][2]
zoom = 'off'
background_color = 'gray'
full_res = 'off'
fig = plt.figure(figsize=(24, 16))
fig.suptitle('blob number ' + str(blobNum + 1), fontsize=20)
fig.patch.set_facecolor(background_color)
gs = gridspec.GridSpec(4, 5, height_ratios=[1, .5, .5, 1])
plotInit(blobNum)
plt.show()

