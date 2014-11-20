import os
import numpy as np
import cv2
import glob
import pandas as pd

# Change the following line to the full path of the directory where your images
# are:
imdir = '/path/to/directory/containing/images/'
# Find all the files in that directory with a .jpeg extension, return a sorted
# list of files
imglist = sorted(glob.glob(imdir+'*.jpeg'))

# Set to false if you do not want to spawn a window showing tracker progress
show_tracks = True

# Size of images to be tracked (note that this is (height, width))
im_size = (600,800)

# Threshold values for tracking, out of a max value of 255 -- set these values
# to just below the pixel values of the thing you want to track; in the example
# video the abdomen was a significantly brighter marker than the coil marker.
abdo_threshval = 200
mark_threshval = 150

# ABDOMEN REGION OF INTEREST
# Define the image
ab_mask = np.zeros(im_size, dtype=np.uint8)

# Define polygon region of interest for the abdomen
# Edit as necessary
ab_center = (400,270)
ab_rmax = (160,160)
ab_rmin = (130,130)
ab_angle = 180
ab_ang_min = -45
ab_ang_max = 45
ab_degrees_per_segment = 5

outer_line = cv2.ellipse2Poly(ab_center, ab_rmax, ab_angle, ab_ang_min, ab_ang_max,
        ab_degrees_per_segment)
inner_line = cv2.ellipse2Poly(ab_center, ab_rmin, ab_angle, ab_ang_min, ab_ang_max,
        ab_degrees_per_segment)
ab_roi_poly = np.append(outer_line, inner_line[::-1]).reshape(-1,2)
cv2.fillConvexPoly(ab_mask, ab_roi_poly, 1)

# MARKER REGION OF INTEREST
# Define polygon region of interest for the abdomen
# Edit as necessary
ma_mask = np.zeros(im_size, dtype=np.uint8)
ma_center = (400,300)
ma_rmax = (270,270)
ma_rmin = (250,250)
ma_angle = 0
ma_ang_min = -45
ma_ang_max = 45
ma_degrees_per_segment = 5

outer_line = cv2.ellipse2Poly(ma_center, ma_rmax, ma_angle, ma_ang_min, ma_ang_max,
        ma_degrees_per_segment)
inner_line = cv2.ellipse2Poly(ma_center, ma_rmin, ma_angle, ma_ang_min, ma_ang_max,
        ma_degrees_per_segment)
ma_roi_poly = np.append(outer_line, inner_line[::-1]).reshape(-1,2)
cv2.fillConvexPoly(ma_mask, ma_roi_poly, 1)


tracks = []

for img_path in imglist:
    # Strip out the frame file, remove the '.jpeg' extension
    frame = os.path.basename(img_path)[:-5]
    # Read in the raw image (we'll use this for display later)
    raw = cv2.imread(img_path)
    # Convert raw image to grayscale
    img = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)

    # Make two new images for analysis by masking the grayscale image with the
    # masks we made for the abdomen and the marker
    abdo_img = cv2.bitwise_and(img,img, mask=ab_mask)
    mark_img = cv2.bitwise_and(img,img, mask=ma_mask)

    # Threshold the abdomen image using the abdoment threshold value, obtaining
    # a binary image
    abdo_thresh = cv2.threshold(abdo_img, abdo_threshval, maxval=1,
            type=cv2.THRESH_BINARY)[1]

    # Calculate the moments of the image -- note that this will only work if the
    # abdomen marker is the ONLY thing in the thresholded ROI
    abdo_moments = cv2.moments(abdo_thresh)
    abdo_centroid_x = abdo_moments['m10']/abdo_moments['m00']
    abdo_centroid_y = abdo_moments['m01']/abdo_moments['m00']

    # Go through the same process for the coil marker
    mark_thresh = cv2.threshold(mark_img, mark_threshval, maxval=1,
            type=cv2.THRESH_BINARY)[1]
    mark_moments = cv2.moments(mark_thresh)
    mark_centroid_x = mark_moments['m10']/mark_moments['m00']
    mark_centroid_y = mark_moments['m01']/mark_moments['m00']


    # Append a dictionary to the tracks list containing the relevant info --
    # we'll turn this into a Pandas dataframe at the end.
    tracks.append({'frame': frame, 'abdo_x': abdo_centroid_x,
        'abdo_y': abdo_centroid_y, 'marker_x': mark_centroid_x,
        'marker_y': mark_centroid_y})

    # If you don't need to see the process happening, feel free to set
    # show_tracks to False at the beginning of the script.
    if show_tracks:

        # Draw a circle at the abdomen centroid on the raw image
        cv2.circle(raw, (int(abdo_centroid_x), int(abdo_centroid_y)), 5,
                (255,0,0))
        # Draw the abdoment region of interest
        cv2.polylines(raw, [ab_roi_poly], True, (255,0,0), 2)
        # Mark the marker centroid
        cv2.circle(raw, (int(mark_centroid_x), int(mark_centroid_y)), 5,
                (0,0,255))
        # Draw the marker ROI
        cv2.polylines(raw, [ma_roi_poly], True, (0,0,255), 2)
        # Show the image in a window
        cv2.imshow('Raw Image', raw)
        # Wait one millisecond before going onto the next frame (increase this
        # value if you want it to go slower, or change it to zero if you want to
        # step through the video frame by frame to check progress
        cv2.waitKey(1)

# Assemble a data frame from the tracks using Pandas
tracks = pd.DataFrame(tracks)

# Uncomment this line to save the data to a CSV file.
#tracks.to_csv('track_data.csv', index=False)
