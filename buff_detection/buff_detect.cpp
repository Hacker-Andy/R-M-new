/****************************************************************************
 *  Copyright (C) 2019 cz.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see <http://www.gnu.org/licenses/>.
 ***************************************************************************/
#include "buff_detect.h"

bool BuffDetector::DetectBuff(Mat& img, OtherParam other_param)
{

    //    GaussianBlur(img, img, Size(3,3),0);
    //    // **预处理** -图像进行相应颜色的二值化
    //    points_2d.clear();
    //    vector<cv::Mat> bgr;
    //    split(img, bgr);
        Mat result_img;
    //    if(color_ != 0)
    //    {
    //        subtract(bgr[2], bgr[1], result_img);
    //    }else
    //    {
    //        subtract(bgr[0], bgr[2], result_img);
    //    }
        Mat binary_color_img;
    //    Mat src = img;
        if(img.empty() || img.channels() != 3){
           cout<<"src.channels() != 3"<<endl;
           return false;
        }
        Mat gray,gray_binary,tempBinary;
        cvtColor(img,gray,COLOR_BGR2GRAY);
        threshold(gray,gray_binary,color_th_,255,THRESH_BINARY);
        // 颜色阈值分割
       Mat imgHSV;
       cvtColor(img,imgHSV,COLOR_BGR2HSV);
       int mode=3;
       if(mode == 3 || mode == 5 || mode == 7){
           Mat temp;
           inRange(imgHSV,Scalar(0,60,80),Scalar(25,255,255),temp);
           inRange(imgHSV,Scalar(156,60,80),Scalar(181,255,255),tempBinary);
           tempBinary = temp | tempBinary;
       }else if(mode == 4 || mode == 6 || mode == 8){
           inRange(imgHSV,Scalar(35,46,80),Scalar(99,255,255),tempBinary);
       }else{
           return false;
       }
    //   imshow("tempBinary",tempBinary);
       dilate(tempBinary,tempBinary,getStructuringElement(MORPH_RECT,Size(3,3)));
       // mask 操作
       binary_color_img = tempBinary & gray_binary;
#ifdef TEST_OTSU
    double th = threshold(result_img, binary_color_img, 50, 255, CV_THRESH_BINARY|CV_THRESH_OTSU);
    if(th-10>0)
        threshold(result_img, binary_color_img, th-10, 255, CV_THRESH_BINARY);
#endif
#ifndef TEST_OTSU
//    threshold(result_img, binary_color_img, color_th_, 255, CV_THRESH_BINARY);
#endif
    //        Mat element = getStructuringElement(MORPH_RECT, Size(5,5));
    //        morphologyEx(binary_color_img,binary_color_img,MORPH_CLOSE,element);
    //        dilate(img, img, element);
#ifdef DEBUG_BUFF_DETECT
    imshow("mask", binary_color_img);
#endif

#ifdef TEST_OTSU
    if(th < 20)
        return 0;
#endif
    // **寻找击打矩形目标** -通过几何关系
    // 寻找识别物体并分类到object
    vector<Object> vec_target;
    vector<Rect> vec_color_rect;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(binary_color_img,contours,hierarchy,CV_RETR_CCOMP,CHAIN_APPROX_NONE);
    for(size_t i=0; i < contours.size();i++)
    {
//        cout<<"findContours"<<endl;

        // 用于寻找小轮廓，没有父轮廓的跳过, 以及不满足6点拟合椭圆
        if(hierarchy[i][3]<0 || contours[i].size() < 6 || contours[static_cast<uint>(hierarchy[i][3])].size() < 6)
            continue;

        // 小轮廓面积条件
        double small_rect_area = contourArea(contours[i]);
        double small_rect_length = arcLength(contours[i],true);
        if(small_rect_length < 10)
            continue;
        // 用于超预测时比例扩展时矩形的判断
        Rect rect = boundingRect(contours[static_cast<uint>(hierarchy[i][3])]);
        vec_color_rect.push_back(rect);

        if(small_rect_area < 200)
            continue;
        // 大轮廓面积条件
        double big_rect_area = contourArea(contours[static_cast<uint>(hierarchy[i][3])]);
        double big_rect_length = arcLength(contours[static_cast<uint>(hierarchy[i][3])],true);
        if(big_rect_area < 300)
            continue;
        if(big_rect_length < 50)
            continue;
        // 能量机关扇叶进行拟合
        Object object;
#ifdef FUSION_MINAREA_ELLIPASE

        object.small_rect_=fitEllipse(contours[i]);
        object.big_rect_ = fitEllipse(contours[static_cast<uint>(hierarchy[i][3])]);
#else
        object.small_rect_=minAreaRect(contours[i]);
        object.big_rect_ = minAreaRect(contours[static_cast<uint>(hierarchy[i][3])]);
#endif

#ifdef DEBUG_DRAW_CONTOURS
        Point2f small_point_tmp[4];
        object.small_rect_.points(small_point_tmp);
        Point2f big_point_tmp[4];
        object.big_rect_.points(big_point_tmp);
        for(int k=0;k<4;k++)
        {
            line(img, small_point_tmp[k],small_point_tmp[(k+1)%4], Scalar(0, 255, 255), 1);
            line(img, big_point_tmp[k],big_point_tmp[(k+1)%4], Scalar(0, 0, 255), 1);
        }

#endif
#ifdef FUSION_MINAREA_ELLIPASE
        object.diff_angle=fabsf(object.big_rect_.angle-object.small_rect_.angle);

        if(object.small_rect_.size.height/object.small_rect_.size.width < 3)
        {
//            cout<<"object.big_rect_.angle"<<object.big_rect_.angle<<endl;
//            cout<<"object.small_rect_.angle"<<object.small_rect_.angle<<endl;
//            cout<<"object.diff_angle"<<object.diff_angle<<endl;

            if(object.diff_angle<100 && object.diff_angle>80)
            {
#endif
#ifndef  FUSION_MINAREA_ELLIPASE
                float small_rect_size_ratio;
                if(object.small_rect_.size.width > object.small_rect_.size.height)
                {
                    small_rect_size_ratio = object.small_rect_.size.width/object.small_rect_.size.height;
                }else {
                    small_rect_size_ratio = object.small_rect_.size.height/object.small_rect_.size.width;
                }
#endif

#ifdef FUSION_MINAREA_ELLIPASE
                float small_rect_size_ratio;
                small_rect_size_ratio = object.small_rect_.size.height/object.small_rect_.size.width;
#endif
                // 根据轮廓面积进行判断扇叶类型

                double area_ratio = area_ratio_/100;
                if(small_rect_area * 12 >big_rect_area && small_rect_area* area_ratio<big_rect_area
                        && small_rect_size_ratio > 1 && small_rect_size_ratio < 3.0f)
                {
                    object.type_ = ACTION;  // 已经激活类型
                }else if(small_rect_area * area_ratio>=big_rect_area && small_rect_area *2 < big_rect_area
                         && small_rect_size_ratio > 1 && small_rect_size_ratio < 3.0f)
                {

                    // 更新世界坐标系顺序
                    object.type_ = INACTION;    // 未激活类型
                    object.UpdateOrder();
                    object.KnowYourself(binary_color_img);
                    vec_target.push_back(object);
                }else
                {
                    object.type_ = UNKOWN;    // 未激活类型
                }
#ifdef AREA_LENGTH_ANGLE
                switch (AREA_LENGTH_ANGLE)
                {
                case 1:
                {
                    double multiple_area=fabs(big_rect_area/small_rect_area);
                    putText(img, to_string(multiple_area), Point2f(50,50)+ object.small_rect_.center, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255));
                }break;
                case 2:
                {
                    double multiple_length=fabs(big_rect_length/small_rect_length);
                    putText(img, to_string(multiple_length), Point2f(50,50)+ object.small_rect_.center, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255));
                }break;
                case 3:
                {
                    putText(img, to_string(object.diff_angle), Point2f(-20,-20)+ object.small_rect_.center, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255));
                }break;
                }
#endif

#ifdef FUSION_MINAREA_ELLIPASE
            }
        }
#endif
    }
    // 遍历所有结果并处理\选择需要击打的目标
    Object final_target;
    bool find_flag = false;
    float diff_angle = 1e8;
    // 你需要击打的能量机关类型 1(true)击打未激活 0(false)击打激活
    for(size_t i=0; i < vec_target.size(); i++)
    {
        Object object_tmp = vec_target.at(i);
        // 普通模式击打未激活机关
        if(object_tmp.type_ == INACTION){
            find_flag = true;
            float ang = fabs(vec_target[i].diff_angle-90.0f);
            if(ang < diff_angle)
            {
                final_target = vec_target.at(i);
                diff_angle = ang;
            }
            putText(img, "final_target", Point2f(10,-50)+ final_target.small_rect_.center, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255));

            Point2f buff_offset = Point2f(buff_offset_x_ - 100, 100 - buff_offset_y_);
            vector<Point2f> vec_points_2d_tmp;
            for(size_t k=0; k < 4; k++)
            {
                vec_points_2d_tmp.push_back(final_target.points_2d_.at(k) + buff_offset);
            }
            points_2d = vec_points_2d_tmp;
            buff_angle_ = final_target.angle_;
#ifdef DEBUG_PUT_TEST_ANGLE
            for(size_t j = 0; j < 4; j++)
            {
                putText(img, to_string(j), Point2f(5,5)+ final_target.points_2d_[j], FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255));
            }
#endif
        }
    }
    if(find_flag){


#ifdef DEBUG_PUT_TEST_TARGET
//        putText(img, "<<---attack here"/*to_string(object_tmp.angle_)*/, Point2f(5,5)+ final_target.small_rect_.center, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255));
#endif
#ifdef DEBUG_DRAW_TARGET
        final_target.DrawTarget(img);
#endif
    }
    return find_flag;
}




int BuffDetector::BuffDetectTask(Mat& img, OtherParam other_param)
{
    color_ = other_param.color;
//    gimbal=other_param.gimbal_data + (pitch_offset-2000)/100;
    buff_offset_x_ = begin_offset_x_ + 3*other_param.buff_offset_x;
    buff_offset_y_ = begin_offset_y_ + 3*other_param.buff_offset_y;
    bool find_flag = DetectBuff(img,other_param);
    int command = 0;
    bool is_change = false;
    float speed,time;
    float fake_speed,fake_time;
    if(find_flag)
    {
        find_cnt ++;
        int fake_direction_tmp = 0;
        float fake_speed=0;
        //15 -》3

        if(find_cnt)
        {
            //            direc`tion_tmp = getDirection(buff_angle_);
            direction_tmp = getSimpleDirectionAndSpeed(buff_angle_, speed,time);
            SPEED_C.push_back(speed);
            TIME_C.push_back(time);
//            cout<<"This is the fucking speed"<<SPEED_C.at(CNT)<<endl;
//            cout<<"This is the fucking TIMEE"<<TIME_C.at(CNT)<<endl;
//            cout<<"This is the fucking number"<<CNT<<endl;
            TIME_C.at(CNT);
            if(CNT<50){
                CNT++;
                return 0;
            }

        }
//        fake_direction_tmp = getSimpleDirectionAndSpeed(buff_angle_, speed,time);
        
        Point2f world_offset;
        #define DIRECTION_FILTER
#ifdef DIRECTION_FILTER
        float world_offset_x = world_offset_x_ - 500;
        float world_offset_y = 800 - pow((640000 - pow(world_offset_x, 2)), 0.5);
        float pre_angle;
        if(direction_tmp == 1)  // shun
        {
            world_offset = Point2f(-world_offset_x, -world_offset_y);
            pre_angle = atan(world_offset_x/(800-world_offset_y));
        }
        else if(direction_tmp == -1)// ni
        {
            world_offset = Point2f(world_offset_x, -world_offset_y);
            pre_angle = -atan(world_offset_x/(800-world_offset_y));
        }
        else
        {
            world_offset = Point2f(0, 0);
            pre_angle = 0;
        }
        //        cout << "direction " << direction_tmp << endl;
        float PreAngle=0;
        PreAngle = getPredictAngle();
        cout<<"This is fucking Angle ======================================="<<PreAngle<<"============="<<endl;

#else
        world_offset = Point2f(world_offset_x_ - 500, world_offset_y_  - 500);
#endif
        solve_angle_long_.Generate3DPoints(2, world_offset);
        solve_angle_long_.getBuffAngle(1,points_2d, BULLET_SPEED
                                       , buff_angle_, pre_angle, gimbal
                                       , angle_x_, angle_y_, distance_);

#ifdef LINER_OFFSET_PITCH
        angle_y_ = angle_y_ + solve_angle_long_.buff_h * (pitch_offset-2000)/160000;
#endif


    }
    //    attack.run(find_flag,angle_x_,angle_y_,target_size,gimbal,direction_tmp);
    command = auto_control.run(angle_x_, angle_y_, find_flag, diff_angle_);

    //    pro_predict_mode_flag = auto_control.GetProPredictFlag();

#ifdef DEBUG_PLOT //0紫 1橙
//    w_->addPoint((auto_control.fire_cnt%2==0)*100+100, 0);
//    w_->addPoint(solve_angle_long_.buff_h, 0);

    w_->plot();
#endif
    return command;
}

int BuffDetector::getSimpleDirectionAndSpeed(float angle, float &speed, float &time)
{
    diff_angle_ = angle - last_angle_;
    last_angle_ = angle;

    if (Time_flag == 1){
        t0 = cv::getTickCount();
        Time_flag = 0;
    }

    if (LT%2==0){

        if (t1 != 0)
        {
            t2 =cv::getTickCount();
            float timediff = ((t2 - t1)/cv::getTickFrequency())*0.1; // 单位 s
            speed = diff_angle_/(timediff*57.3);
            //std::cout<<"This is timediff:: "<<timediff<<" =============This is anglediff::"<<diff_angle_<<std::endl;
            //std::cout<<"The fucking speed:::"<<speed<<std::endl;
            //std::cout<<"The fucking Timer(s):::"<<((t2-t0)/cv::getTickFrequency())<<std::endl;
//            std::cout<<"cv Freq:::"<<cv::getTickFrequency()<<std::endl;
            time = ((t2-t0)/cv::getTickFrequency());
            ofstream TransDataSpeed("./Speed.txt",ios::app); //存放我们发送数据的文件
            ofstream TransDataTime("./Time.txt",ios::app); //存放我们发送数据的文件
            ofstream TransDataTimeDiff("./TimeDiff.txt",ios::app); //存放我们发送数据的文件
            ofstream TransDataAngleDiff("./AngleDiff.txt",ios::app); //存放我们发送数据的文件

            TransDataSpeed<<speed<<endl;
            TransDataTime<<((t2-t0)/cv::getTickFrequency())<<endl;
            TransDataTimeDiff<<timediff<<endl;
            TransDataAngleDiff<<diff_angle_<<endl;

            TransDataSpeed.close();
            TransDataTime.close();
            TransDataTimeDiff.close();
            TransDataAngleDiff.close();

        }
        t1 = cv::getTickCount();
    }
    LT++;
    if(fabs(diff_angle_) < 10 && fabs(diff_angle_) > 1e-6)
    {
        d_angle_ = (1 - r) * d_angle_ + r * diff_angle_;
    }
    if(d_angle_ > 2)
        return 1;
    else if(d_angle_ < -2)
        return -1;
    else
        return 0;
}

void Object::UpdateOrder()
{
    points_2d_.clear();
#ifdef FUSION_MINAREA_ELLIPASE
    Point2f points[4];
    small_rect_.points(points);
    Point2f point_up_center = (points[0] + points[1])/2;
    Point2f point_down_center = (points[2] + points[3])/2;
    double up_distance = Point_distance(point_up_center, big_rect_.center);
    double down_distance = Point_distance(point_down_center, big_rect_.center);
    if(up_distance > down_distance)
    {
        angle_ = small_rect_.angle;
        points_2d_.push_back(points[0]);points_2d_.push_back(points[1]);
        points_2d_.push_back(points[2]);points_2d_.push_back(points[3]);
    }else
    {
        angle_ = small_rect_.angle + 180;
        points_2d_.push_back(points[2]);points_2d_.push_back(points[3]);
        points_2d_.push_back(points[0]);points_2d_.push_back(points[1]);
    }
#else
    float width = small_rect_.size.width;
    float height = small_rect_.size.height;
    Point2f points[4];
    small_rect_.points(points);
    if(width >= height)
    {
        Point2f point_up_center = (points[0] + points[3])/2;
        Point2f point_down_center = (points[1] + points[2])/2;
        float up_distance = Point_distance(point_up_center, big_rect_.center);
        float down_distance = Point_distance(point_down_center, big_rect_.center);
        if(up_distance <= down_distance)
        {
            angle_ = 90 - small_rect_.angle;
            points_2d_.push_back(points[1]);points_2d_.push_back(points[2]);
            points_2d_.push_back(points[3]);points_2d_.push_back(points[0]);

        }else
        {
            angle_ = 270 - small_rect_.angle;
            points_2d_.push_back(points[3]);points_2d_.push_back(points[0]);
            points_2d_.push_back(points[1]);points_2d_.push_back(points[2]);
        }
    }else
    {
        Point2f point_up_center = (points[0] + points[1])/2;
        Point2f point_down_center = (points[2] + points[3])/2;
        float up_distance = Point_distance(point_up_center, big_rect_.center);
        float down_distance = Point_distance(point_down_center, big_rect_.center);
        if(up_distance <= down_distance)
        {
            angle_ = - small_rect_.angle;
            points_2d_.push_back(points[2]);points_2d_.push_back(points[3]);
            points_2d_.push_back(points[0]);points_2d_.push_back(points[1]);

        }else
        {
            angle_ = 180 - small_rect_.angle;
            points_2d_.push_back(points[0]);points_2d_.push_back(points[1]);
            points_2d_.push_back(points[2]);points_2d_.push_back(points[3]);
        }
    }
#endif
}


int GetRectIntensity(const Mat &img, Rect rect){
    if(rect.width < 1 || rect.height < 1 || rect.x < 1 || rect.y < 1
            || rect.width + rect.x > img.cols || rect.height + rect.y > img.rows)
        return 255;
    Mat roi = img(Range(rect.y, rect.y + rect.height), Range(rect.x, rect.x + rect.width) );
    //        imshow("roi ", roi);
    int average_intensity = static_cast<int>(mean(roi).val[0]);
    return average_intensity;
}

void Object::KnowYourself(Mat &img)
{
    Point2f vector_height = points_2d_.at(0) - points_2d_.at(3);
    //    vector_height = Point2f(vector_height.x * 0.5 , vector_height.y * 0.5);
    Point left_center = points_2d_.at(3) - vector_height;
    Point right_center = points_2d_.at(2) - vector_height;
    //        circle(img, left_center, 3, Scalar(255), -1);
    //        circle(img, right_center, 3, Scalar(255), 1);

    int width = 5;
    int height = 5;

    Point left1 = Point(left_center.x - width, left_center.y - height);
    Point left2 = Point(left_center.x + width, left_center.y + height);

    Point right1 = Point(right_center.x - width, right_center.y - height);
    Point right2 = Point(right_center.x + width, right_center.y + height);

    Rect left_rect(left1, left2);
    Rect right_rect(right1, right2);

    //    rectangle(img, left_rect, Scalar(255), 1);
    //    rectangle(img, right_rect, Scalar(255), 1);

    int left_intensity = GetRectIntensity(img, left_rect);
    int right_intensity = GetRectIntensity(img, right_rect);
    if(left_intensity > 10 && right_intensity > 10)
    {
        type_ = ACTION;
    }else{
        type_ = INACTION;
    }
    putText(img, to_string(left_intensity), left_center, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255));
    putText(img, to_string(right_intensity), right_center, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255));
#ifdef IMSHOW_2_ROI
    imshow("test", img);
#endif
}



double Point_distance(Point2f p1,Point2f p2)
{
    double Dis=pow(pow((p1.x-p2.x),2)+pow((p1.y-p2.y),2),0.5);
    return Dis;
}

float BuffDetector::getPredictAngle(){
   //fit sin to get W
   float a=0.785, b=1.884, c = 1.305;
   // int CNT = 50;             //the number of FIT
   float lr = 0.001;   //learning rate
//   float W=0;
//   float T;
   float deltasita=0.0;
   float deltaTime = 0.3;
//   cout<<"This is for Test :: Speed::"<<SPEED_C.at(49)<<endl;
   if(ReFit < 3){
       w=0;
       for(int i = 0; i < 40; i++){
            w = w - lr*(SPEED_C.at(i) - a*sin(b*TIME_C.at(i) + w)*(-a*cos(b*TIME_C.at(i)+w)));
       }
       ReFit ++;
   }
//   SPEED_C.clear();
//   TIME_C.clear();
   //get sita
   deltasita = -a*cos(b*(TIME_C.at(TIME_C.size()-1)+deltaTime) + w)/b + a*cos(b*(TIME_C.size()-1) + w)/b + c*(deltaTime);

   return deltasita;
}
