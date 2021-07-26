#include <iostream>
#include <iomanip>
#include <string.h>
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv.hpp"


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

Scalar colorTab[] = {
    Scalar(0, 0, 255),
    Scalar(0,255,0),
    Scalar(255,100,100),
    Scalar(255,0,255),
    Scalar(0,255,255),
    Scalar(255, 0, 0)
};
int object_heigh = 580;
int object_width = 190;
int object_x = 160;
int object_y = 1300;
int garmin_heigh = 225;
int garmin_width = 50;
int garmin_x = 220;
int garmin_y = 2105;
int two_object_heigh = 1185;
int two_object_width = 180;

void split_image(Mat scene_image, vector<Mat> &images){
    images.push_back(scene_image(Range(0, scene_image.rows/3 -1), Range(0, scene_image.cols/2 -1)));
    images.push_back(scene_image(Range(0, scene_image.rows/3 -1), Range(scene_image.cols/2, scene_image.cols-1)));
    images.push_back(scene_image(Range(scene_image.rows/3, 2*scene_image.rows/3 -1), Range(0, scene_image.cols/2 -1)));
    images.push_back(scene_image(Range(scene_image.rows/3, 2*scene_image.rows/3 -1), Range(scene_image.cols/2, scene_image.cols-1)));
    images.push_back(scene_image(Range(2*scene_image.rows/3, scene_image.rows -1), Range(0, scene_image.cols/2 -1)));
    images.push_back(scene_image(Range(2*scene_image.rows/3, scene_image.rows -1), Range(scene_image.cols/2, scene_image.cols-1)));
}
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
Mat surf(Mat scene_image, Mat object_image, const float ratio_threshold, int K){
    // Mask
    // Mat mask = Mat::ones(scene_image.size(), CV_8U);
    // Mat roi(mask, cv::Rect(object_x, object_y, two_object_width, two_object_heigh));
    // roi = Scalar(0);

    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int hessian_threshold = 400;
    Ptr<SURF> detector = SURF::create();
    vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;
    detector->detectAndCompute(object_image, noArray(), keypoints_object, descriptors_object);
    detector->detectAndCompute(scene_image, noArray(), keypoints_scene, descriptors_scene);

    /* Cluster */
    vector<Point2f> keypoints_scene_points_;
    for(int i=0; i<keypoints_scene.size(); i++){
        keypoints_scene_points_.push_back(keypoints_scene[i].pt);
    }
    Mat keypoints_scene_points(keypoints_scene_points_), labels;
    vector<Point2f> centers;
    double compactness = kmeans(keypoints_scene_points, K, labels, 
        TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0), 3, KMEANS_RANDOM_CENTERS, centers);
    
    Mat *descriptors_scene_clusters = new Mat[K];
    for(int i=0; i<K; i++){
        cout<<i<<endl;
        descriptors_scene_clusters[i] = Mat(descriptors_scene.rows, descriptors_scene.cols, CV_32F, Scalar(0));
        cout<<i<<endl;
    }

    Mat kmeans_image;
    cvtColor(scene_image, kmeans_image, COLOR_GRAY2RGB);
    for(int i = 0; i < keypoints_scene_points_.size(); i++){
        int clusterIdx = labels.at<int>(i);
        Point2f ipt = keypoints_scene_points.at<Point2f>(i);
        descriptors_scene.row(i).copyTo(descriptors_scene_clusters[clusterIdx].row(i));
        circle(kmeans_image , ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA);
    }

    /*  Debug   */

    // cout<<keypoints_scene_points_.size()<<" "<<descriptors_scene.size()<<" "<<descriptors_scene_clusters[0].size()<<endl;
    // for(int j=0; j<descriptors_scene.cols; j++){
    //     cout<<j<<":"<<descriptors_scene.at<float>(0, j)<<" "<<descriptors_scene_clusters[0].at<float>(0, j)<<endl;
    // }
    // Mat diff = descriptors_scene != descriptors_scene_clusters[0];
    // bool eq = countNonZero(diff) == 0;
    // cout<<eq<<" "<<countNonZero(diff)<<" "<<descriptors_scene.size()<<endl;

    // int sum1=0, sum2=0, sum3=0;
    // float tmp1, tmp2, tmp3;
    // for(int i = 0; i < keypoints_scene_points_.size(); i++){
    //     Point2f ipt = keypoints_scene_points.at<Point2f>(i);
    //     tmp1 = descriptors_scene.at<float>(ipt);
    //     tmp2 = descriptors_scene_clusters[0].at<float>(ipt);
    //     tmp3 = descriptors_scene_clusters[1].at<float>(ipt);
    //     if(tmp1 != tmp2 && tmp1 != tmp3){
    //         cout<<"Point["<<i<<"]: "<<tmp1<<" "<<tmp2<<" "<<tmp3<<endl;
    //     }
        
    //     if(descriptors_scene.at<float>(ipt)!=0.0)
    //         sum1++;
    //     if(descriptors_scene_clusters[0].at<float>(ipt)!=0.0)
    //         sum2++;
    //     if(descriptors_scene_clusters[1].at<float>(ipt)!=0.0)
    //         sum3++;
    // }
    // cout<<sum1<<" "<<sum2<<" "<<sum3<<endl;
    /*  --- */

    for(int i = 0; i < (int)centers.size(); ++i){
        Point2f ipt = centers[i];
        circle(kmeans_image, ipt, 500, colorTab[i], 5, LINE_AA );
    }
    imshow("kmeans", kmeans_image);
    waitKey();

    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    vector<DMatch> all_good_matches;
    vector<Point2f> mask_points;
    Mat matches_image;
    for(int i=0; i<K; i++){
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        vector<vector<DMatch>> knn_matches;
        if(descriptors_object.empty()){
            cout<<"descriptors_object is empty!\n";
        } 
        if(descriptors_scene_clusters[i].empty()){
            cout<<"descriptors_scene_clusters["<<i<<"] is empty!\n";
            continue;
        }

        
        matcher->knnMatch(descriptors_object, descriptors_scene_clusters[i], knn_matches, 2);

        //-- Filter matches using the Lowe's ratio test
        vector<DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++){
            if(knn_matches[i][0].distance < ratio_threshold * knn_matches[i][1].distance)
                good_matches.push_back(knn_matches[i][0]);
        }
        cout<<"# of good matches in cluster["<<i<<"]: "<<good_matches.size()<<"/"<<knn_matches.size()<<endl;
        all_good_matches.insert(all_good_matches.end(), good_matches.begin(), good_matches.end());
        

        drawMatches(object_image, keypoints_object, scene_image, keypoints_scene, good_matches, matches_image, Scalar::all(-1),
                            Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        // circle(matches_image, centers[i], 1000, colorTab[i], 5, LINE_AA);

        // Homography at least 4 points
        if(good_matches.size()>=4){
            vector<Point2f> obj;
            vector<Point2f> scene;
            for(size_t i = 0; i < good_matches.size(); i++){
                //-- Get the keypoints from the good matches
                obj.push_back(keypoints_object[ good_matches[i].queryIdx ].pt);
                scene.push_back(keypoints_scene[ good_matches[i].trainIdx ].pt);
            }

            Mat H = findHomography(obj, scene, RANSAC);
            if(!H.empty()){
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
                mask_points.push_back(scene_corners[0]);

                //-- Draw lines between the corners (the mapped object in the scene - image_2 )
                line(matches_image, scene_corners[0] + Point2f((float)object_image.cols, 0),
                    scene_corners[1] + Point2f((float)object_image.cols, 0), Scalar(0, 255, 0), 4);
                line(matches_image, scene_corners[1] + Point2f((float)object_image.cols, 0),
                    scene_corners[2] + Point2f((float)object_image.cols, 0), Scalar( 0, 255, 0), 4);
                line(matches_image, scene_corners[2] + Point2f((float)object_image.cols, 0),
                    scene_corners[3] + Point2f((float)object_image.cols, 0), Scalar( 0, 255, 0), 4);
                line(matches_image, scene_corners[3] + Point2f((float)object_image.cols, 0),
                    scene_corners[0] + Point2f((float)object_image.cols, 0), Scalar( 0, 255, 0), 4);
            
                string title = "Cluster " + to_string(i);
                imshow(title, matches_image);
                waitKey();
            }else{
                cout<<"Cluster["<<i<<"]'s homography is empty!\n";
            }

        }
    }
    drawMatches(object_image, keypoints_object, scene_image, keypoints_scene, all_good_matches, matches_image, Scalar::all(-1),
                            Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    delete [] descriptors_scene_clusters;
    return matches_image;
}
int main(const int argc, const char *argv[]){
    if (argc<5){
        cout << "Not enough parameters" << endl;
        cout << "Usage: " << argv[0]<< " -i <scene image> "
                                    << " -t <test image>"
                                    << " -r <ratio threshold>" 
                                    << " -a <matching algorithm>"
                                    << " -k <number of clusters>"<< endl;
        return -1;
    }
    Mat scene_image;
    Mat test_image;
    char algo[10];
    float ratio_threshold;
    int nsubplot = 1;
    int K = 1;

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
        else if(strcmp(argv[i], "-k")==0){
            K = atoi(argv[i+1]);
        }
    }
    Rect object_size(object_x, object_y, object_width, object_heigh);
    Mat object_image = scene_image(object_size);

    Rect two_object_size(object_x, object_y, two_object_width, two_object_heigh);
    Mat two_object_image = scene_image(two_object_size);

    Rect garmin_size(garmin_x, garmin_y, garmin_width, garmin_heigh);
    Mat garmin_image = scene_image(garmin_size);

    if(test_image.empty())
        test_image = two_object_image;

    //  Images position:
    //  0   1
    //  2   3
    //  4   5
    // vector<Mat> scene_images;
    // split_image(scene_image, scene_images);

    vector<Mat> matches;
    if(strcmp(algo, "surf")==0){
        imshow("Surf", surf(scene_image, test_image, ratio_threshold, K));
        waitKey();
        // for(int i=0; i<scene_images.size(); i++){
        //     matches.push_back(surf(scene_images[i], test_image, ratio_threshold));
        // }
        // subplot("Surf", matches);
        matches.clear();
    }
    else if(strcmp(algo, "sift")==0){
        Mat result = sift(scene_image, test_image, ratio_threshold);
        imshow("Sift", result);
        waitKey();
        // for(int i=0; i<scene_images.size(); i++){
        //     matches.push_back(sift(scene_images[i], test_image,  ratio_threshold));
        // }
        // subplot("Sift", matches);
        matches.clear();
    }
    return 0;
}