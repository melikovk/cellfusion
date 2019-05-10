import os
import sys
import glob
import json
import dlib

# In this example we are going to train a face detector based on the small
# faces dataset in the examples/faces directory.  This means you need to supply
# the path to this faces folder as a command line argument so we will know
# where it is.

# if len(sys.argv) != 2:
#     print(
#         "Give the path to the examples/faces directory as the argument to this "
#         "program. For example, if you are in the python_examples folder then "
#         "execute this program by running:\n"
#         "    ./train_object_detector.py ../examples/faces")
#     exit()
# faces_folder = sys.argv[1]


# Now let's do the training.  The train_simple_object_detector() function has a
# bunch of options, all of which come with reasonable default values.  The next
# few lines goes over some of these options.
options = dlib.simple_object_detector_training_options()
# Since faces are left/right symmetric we can tell the trainer to train a
# symmetric detector.  This helps it get the most value out of the training
# data.
options.add_left_right_image_flips = True
# The trainer is a kind of support vector machine and therefore has the usual
# SVM C parameter.  In general, a bigger C encourages it to fit the training
# data better but might lead to overfitting.  You must find the best C value
# empirically by checking how well the trained detector works on a test set of
# images you haven't trained on.  Don't just leave the value set at 5.  Try a
# few different C values and see what works best for your data.
options.C = 5
# Tell the code how many CPU cores your computer has for the fastest training.
options.num_threads = 4
options.be_verbose = True
options.detection_window_size = 80*80
print(options)
# Finally, note that you don't have to use the XML based input to
# train_simple_object_detector().  If you have already loaded your training
# images and bounding boxes for the objects then you can call it as shown
# below.

imgnames = glob.glob("/home/fast/Automate/20x/nuclei/dlib/train/*.png")
# You just need to put your images into a list.
images = []
boxes = []
for img in imgnames:
    images.append(dlib.load_grayscale_image(img))
    with open(img[:-4]+'.json', 'r') as f:
        imgboxes = map(lambda x: dlib.rectangle(left=x[0], top=x[1],
                                                right=x[0]+x[2], bottom=x[1]+x[3]),
                       json.load(f))
        boxes.append(list(imgboxes))

print(len(boxes), len(images))

detector = dlib.train_simple_object_detector(images, boxes, options)

# We could save this detector to disk by uncommenting the following.
detector.save('detector.svm')

# Now let's look at its HOG filter!
# We can look at the HOG filter we learned.  It should look like a face.  Neat!
win_det = dlib.image_window()
win_det.set_image(detector)
dlib.hit_enter_to_continue()

# Note that you don't have to use the XML based input to
# test_simple_object_detector().  If you have already loaded your training
# images and bounding boxes for the objects then you can call it as shown
# below.
print("\nTraining accuracy: {}".format(
    dlib.test_simple_object_detector(images, boxes, detector)))

# Now let's run the detector over the images in the faces folder and display the
# results.

testnames = glob.glob("/home/fast/Automate/20x/nuclei/dlib/test/*.png")
testimages = []
testboxes = []
for img in testnames:
    testimages.append(dlib.load_grayscale_image(img))
    with open(img[:-4]+'.json', 'r') as f:
        imgboxes = map(lambda x: dlib.rectangle(left=x[0], top=x[1],
                                                right=x[0]+x[2], bottom=x[1]+x[3]),
                       json.load(f))
        testboxes.append(list(imgboxes))

idx = 0
print("Showing detections on the images in the faces folder...")
win = dlib.image_window()
print("Processing file: {}".format(f))
img = testimages[idx]
dets = detector(img)
# print(dets)
# print(boxes[0])
print("Number of faces detected: {}".format(len(dets)))
win.clear_overlay()
win.set_image(img)
win.add_overlay(dets)
color = dlib.rgb_pixel(0,255,0)
win.add_overlay(testboxes[idx], color=color)
dlib.hit_enter_to_continue()


# Next, suppose you have trained multiple detectors and you want to run them
# efficiently as a group.  You can do this as follows:
detector1 = dlib.fhog_object_detector("detector.svm")
# In this example we load detector.svm again since it's the only one we have on
# hand. But in general it would be a different detector.
detector2 = dlib.fhog_object_detector("detector.svm")
# make a list of all the detectors you wan to run.  Here we have 2, but you
# could have any number.
detectors = [detector1, detector2]
image = dlib.load_rgb_image(faces_folder + '/2008_002506.jpg')
[boxes, confidences, detector_idxs] = dlib.fhog_object_detector.run_multiple(detectors, image, upsample_num_times=1, adjust_threshold=0.0)
for i in range(len(boxes)):
    print("detector {} found box {} with confidence {}.".format(detector_idxs[i], boxes[i], confidences[i]))
