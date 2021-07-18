#include <iostream>
#include <iomanip>
#include <string.h>
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

void subplot(string title, vector<Mat> matches){

    //  width   - Maximum number of images in a row
    //  heigh   - Maximum number of images in a column
    //  size    - size of one image    
    //  x       - # of cols from input image
    //  y       - # of rows from input image
    int width, heigh, size, x, y;
    float scale;

    if((int)matches.size() <= 0){
        printf("Number of arguments too small....\n");
        return;
    }else if((int)matches.size() > 12){
        printf("Number of arguments too large, can only handle maximally 12 images at a time ...\n");
        return;
    }else if((int)matches.size() == 1){
        width = 1;
        heigh = 1;
        size = 500;
    }else if((int)matches.size() == 2){
        width = 2;
        heigh = 1;
        size = 500;
    }else if((int)matches.size() == 3 || (int)matches.size() == 4){
        width = 2;
        heigh = 2;
        size = 500;
    }else if((int)matches.size() == 5 || (int)matches.size() == 6){
        width = 2;
        heigh = 3;
        size = 500;
    }else if((int)matches.size() == 7 || (int)matches.size() == 8){
        width = 2;
        heigh = 4;
        size = 200;
    }else{
        width = 3;
        heigh = 4;
        size = 150;
    }

    // Create a new 3 channel image
    Mat DispImage = Mat::zeros(Size(100 + size*width, 60 + size*heigh), CV_8UC3);

    for(int i=0, m=20, n=20; !matches.empty(); i++, m+=(20+size)){
        Mat image = *matches.begin();
        matches.erase(matches.begin());
        if(image.empty()){
            cout<<"Invalid image at ."<<i<<"\n";
            return;
        }

        x = image.cols;
        y = image.rows;
        // Find the scaling factor to resize the image
        scale = (float)( (float)max(x, y) / size);

        // Used to Align the images
        if( i % width == 0 && m!= 20) {
            m = 20;
            n+= 20 + size;
        }
        Rect ROI(m, n, (int)( x/scale ), (int)( y/scale ));
        Mat temp; 
        resize(image, temp, Size(ROI.width, ROI.height));
        temp.copyTo(DispImage(ROI));
    }
    imwrite("./result/sub.png", DispImage);
    namedWindow(title, 1);
    imshow(title, DispImage);
    waitKey();
}
Mat sift(Mat scene_image, Mat object_image, const float ratio_threshold){
    Ptr<SIFT> detector = SIFT::create();
    vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;
    detector->detectAndCompute(object_image, noArray(), keypoints_object, descriptors_object);
    detector->detectAndCompute(scene_image, noArray(), keypoints_scene, descriptors_scene); 

    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    vector<vector<DMatch>> knn_matches;
    matcher->knnMatch(descriptors_object, descriptors_scene, knn_matches, 2);
    cout<<"knn matches row size:"<<knn_matches.size()<<endl;

    //-- Filter matches using the Lowe's ratio test
    vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++){
        if (knn_matches[i][0].distance < ratio_threshold * knn_matches[i][1].distance)
            good_matches.push_back(knn_matches[i][0]);
    }
    cout<<"# of good matches:"<<good_matches.size()<<endl;

    //-- Draw matches
    Mat img_matches;
    drawMatches(object_image, keypoints_object, scene_image, keypoints_scene, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    //-- Localize the object
    vector<Point2f> obj;
    vector<Point2f> scene;
    for( size_t i = 0; i < good_matches.size(); i++){
        //-- Get the keypoints from the good matches
        obj.push_back(keypoints_object[ good_matches[i].queryIdx ].pt);
        scene.push_back(keypoints_scene[ good_matches[i].trainIdx ].pt);
    }

    Mat H = findHomography(obj, scene, RANSAC);
    if(H.empty()){
        cout<<"H matrix is empty."<<endl;
        return H;
    }
    //-- Get the corners from the image_1 ( the object to be "detected" )
    vector<Point2f> obj_corners(4);
    vector<Point2f> scene_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f((float)object_image.cols, 0);
    obj_corners[2] = Point2f((float)object_image.cols, (float)object_image.rows);
    obj_corners[3] = Point2f(0, (float)object_image.rows);
    perspectiveTransform(obj_corners, scene_corners, H);

    cout<<"Corner[0]: "<<obj_corners[0]<<" -> "<<scene_corners[0]<<endl;
    cout<<"Corner[1]: "<<obj_corners[1]<<" -> "<<scene_corners[1]<<endl;
    cout<<"Corner[2]: "<<obj_corners[2]<<" -> "<<scene_corners[2]<<endl;
    cout<<"Corner[3]: "<<obj_corners[3]<<" -> "<<scene_corners[3]<<endl;
    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line(img_matches, scene_corners[0] + Point2f((float)object_image.cols, 0),
          scene_corners[1] + Point2f((float)object_image.cols, 0), Scalar(0, 255, 0), 4);
    line(img_matches, scene_corners[1] + Point2f((float)object_image.cols, 0),
        scene_corners[2] + Point2f((float)object_image.cols, 0), Scalar( 0, 255, 0), 4);
    line(img_matches, scene_corners[2] + Point2f((float)object_image.cols, 0),
          scene_corners[3] + Point2f((float)object_image.cols, 0), Scalar( 0, 255, 0), 4);
    line(img_matches, scene_corners[3] + Point2f((float)object_image.cols, 0),
          scene_corners[0] + Point2f((float)object_image.cols, 0), Scalar( 0, 255, 0), 4);

    //-- Show detected matches
    return img_matches;
}
Mat surf(Mat scene_image, Mat object_image, const float ratio_threshold){
    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    Ptr<SURF> detector = SURF::create();
    vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;
    detector->detectAndCompute(object_image, noArray(), keypoints_object, descriptors_object);
    detector->detectAndCompute(scene_image, noArray(), keypoints_scene, descriptors_scene);

    /*  K-means */
    // vector<Point2f> keypoints_scene_points_;
    // for(int i=0; i<keypoints_scene.size(); i++){
    //     keypoints_scene_points_.push_back(keypoints_scene[i].pt);
    // }

    // Mat keypoints_scene_points(keypoints_scene_points_);
    // Mat labels;
    // int K = 6;
    // double compactness = kmeans(keypoints_scene_points, K, labels, 
    //     TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0), 3, KMEANS_RANDOM_CENTERS);
    
    // Scalar colorTab[] = {
    //     Scalar(0, 0, 255),
    //     Scalar(0,255,0),
    //     Scalar(255,100,100),
    //     Scalar(255,0,255),
    //     Scalar(0,255,255),
    //     Scalar(255, 0, 0)
    // };
    // Mat temp;
    // cvtColor(scene_image, temp, COLOR_GRAY2RGB);
    // for(int i = 0; i < keypoints_scene_points_.size(); i++){
    //     int clusterIdx = labels.at<int>(i);
    //     Point2f ipt = keypoints_scene_points.at<Point2f>(i);
    //     circle(temp , ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA);
    // }
    // imshow("k-means", temp);
    // waitKey();
    /*  ------- */

    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    vector<vector<DMatch>> knn_matches;
    matcher->knnMatch(descriptors_object, descriptors_scene, knn_matches, 2);
    cout<<"knn matches row size:"<<knn_matches.size()<<endl;

    //-- Filter matches using the Lowe's ratio test
    vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++){
        if (knn_matches[i][0].distance < ratio_threshold * knn_matches[i][1].distance)
            good_matches.push_back(knn_matches[i][0]);
    }
    cout<<"# of good matches:"<<good_matches.size()<<endl;

    //-- Draw matches
    Mat img_matches;
    drawMatches(object_image, keypoints_object, scene_image, keypoints_scene, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    //-- Localize the object
    vector<Point2f> obj;
    vector<Point2f> scene;
    for( size_t i = 0; i < good_matches.size(); i++){
        //-- Get the keypoints from the good matches
        obj.push_back(keypoints_object[ good_matches[i].queryIdx ].pt);
        scene.push_back(keypoints_scene[ good_matches[i].trainIdx ].pt);
    }

    Mat H = findHomography(obj, scene, RANSAC);
    if(H.empty()){
        cout<<"H matrix is empty."<<endl;
        return H;
    }
    //-- Get the corners from the image_1 ( the object to be "detected" )
    vector<Point2f> obj_corners(4);
    vector<Point2f> scene_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f((float)object_image.cols, 0);
    obj_corners[2] = Point2f((float)object_image.cols, (float)object_image.rows);
    obj_corners[3] = Point2f(0, (float)object_image.rows);
    perspectiveTransform(obj_corners, scene_corners, H);

    cout<<"Corner[0]:"<<obj_corners[0]<<" -> "<<scene_corners[0]<<endl;
    cout<<"Corner[1]:"<<obj_corners[1]<<" -> "<<scene_corners[1]<<endl;
    cout<<"Corner[2]:"<<obj_corners[2]<<" -> "<<scene_corners[2]<<endl;
    cout<<"Corner[3]:"<<obj_corners[3]<<" -> "<<scene_corners[3]<<endl;
    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line(img_matches, scene_corners[0] + Point2f((float)object_image.cols, 0),
          scene_corners[1] + Point2f((float)object_image.cols, 0), Scalar(0, 255, 0), 4);
    line(img_matches, scene_corners[1] + Point2f((float)object_image.cols, 0),
        scene_corners[2] + Point2f((float)object_image.cols, 0), Scalar( 0, 255, 0), 4);
    line(img_matches, scene_corners[2] + Point2f((float)object_image.cols, 0),
          scene_corners[3] + Point2f((float)object_image.cols, 0), Scalar( 0, 255, 0), 4);
    line(img_matches, scene_corners[3] + Point2f((float)object_image.cols, 0),
          scene_corners[0] + Point2f((float)object_image.cols, 0), Scalar( 0, 255, 0), 4);

    return img_matches;
}
int main(const int argc, const char *argv[]){
    if (argc<5){
        cout << "Not enough parameters" << endl;
        cout << "Usage: " << argv[0]<< " -i <scene image> "
                                    << " [-t <test image>]"
                                    << " -r <ratio threshold>" 
                                    << " -a <matching algorithm>"<< endl;
        return -1;
    }
    Mat scene_image;
    Mat test_image;
    char algo[10];
    int object_heigh = 580;
    int object_width = 190;
    int garmin_heigh = 225;
    int garmin_width = 50;
    int two_object_heigh = 1185;
    int two_object_width = 180;
    float ratio_threshold;

    for(int i=1; i<argc; i+=2){
        if(strcmp(argv[i], "-i")==0){
            scene_image = imread(argv[i+1], IMREAD_GRAYSCALE);
            if(scene_image.empty()){
                cout<<"Can not read scene image\n";
                return -1;
            }
        }
        else if(strcmp(argv[i], "-t")==0){
            test_image = imread(argv[i+1], IMREAD_GRAYSCALE);
            if(test_image.empty()){
                cout<<"Can not read test image\n";
                return -1;
            }
        }
        else if(strcmp(argv[i], "-r")==0){
            ratio_threshold = atof(argv[i+1]);
        }
        else if(strcmp(argv[i], "-a")==0){
            strcpy(algo, argv[i+1]);
        }
    }
    Rect object_size(160, 1300, object_width, object_heigh);
    Mat object_image = scene_image(object_size);

    Rect two_object_size(160, 1300, two_object_width, two_object_heigh);
    Mat two_object_image = scene_image(two_object_size);

    Rect garmin_size(220, 2105, garmin_width, garmin_heigh);
    Mat garmin_image = scene_image(garmin_size);

    //  Subimage position:
    //  0   1
    //  2   3
    //  4   5
    Mat scene_image0 = scene_image(Range(0, scene_image.rows/3 -1), Range(0, scene_image.cols/2 -1));
    Mat scene_image1 = scene_image(Range(0, scene_image.rows/3 -1), Range(scene_image.cols/2, scene_image.cols-1));
    Mat scene_image2 = scene_image(Range(scene_image.rows/3, 2*scene_image.rows/3 -1), Range(0, scene_image.cols/2 -1));
    Mat scene_image3 = scene_image(Range(scene_image.rows/3, 2*scene_image.rows/3 -1), Range(scene_image.cols/2, scene_image.cols-1));
    Mat scene_image4 = scene_image(Range(2*scene_image.rows/3, scene_image.rows -1), Range(0, scene_image.cols/2 -1));
    Mat scene_image5 = scene_image(Range(2*scene_image.rows/3, scene_image.rows -1), Range(scene_image.cols/2, scene_image.cols-1));

    // test_image = two_object_image;
    vector<Mat> matches;
    if(strcmp(algo, "surf")==0){
        Mat result = surf(scene_image, test_image, ratio_threshold);
        imshow("Surf", result);
        // imwrite("./result/surf.png", result);
        waitKey();
        matches.push_back(surf(scene_image0, test_image, ratio_threshold));
        matches.push_back(surf(scene_image1, test_image, ratio_threshold));
        matches.push_back(surf(scene_image2, test_image, ratio_threshold));
        matches.push_back(surf(scene_image3, test_image, ratio_threshold));
        matches.push_back(surf(scene_image4, test_image, ratio_threshold));
        matches.push_back(surf(scene_image5, test_image, ratio_threshold));
        subplot("Surf", matches);
        matches.clear();
    }
    else if(strcmp(algo, "sift")==0){
        Mat result = sift(scene_image, test_image, ratio_threshold);
        // imwrite("./result/sift.png", result);
        imshow("Sift", result);
        waitKey();
        matches.push_back(sift(scene_image0, test_image,  ratio_threshold));
        matches.push_back(sift(scene_image1, test_image,  ratio_threshold));
        matches.push_back(sift(scene_image2, test_image,  ratio_threshold));
        matches.push_back(sift(scene_image3, test_image,  ratio_threshold));
        matches.push_back(sift(scene_image4, test_image,  ratio_threshold));
        matches.push_back(sift(scene_image5, test_image,  ratio_threshold));
        subplot("Sift", matches);
        matches.clear();
    }
    return 0;
}