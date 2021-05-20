#include "camera_device.h"
#include "base.h"

using namespace Dahua::GenICam;
using namespace Dahua::Infra;

//#ifndef CANCLE_NNZ

CameraDevice :: CameraDevice()
{
    //discovery device
    //ICameraPtr cameraSptr;
    TVector<ICameraPtr>vCameraPtrList;
    isDiscoverySuccess = systemObj.discovery(vCameraPtrList);
    cameraSptr = vCameraPtrList[0];
    nDeviceNum = vCameraPtrList.size();
    status = cameraSptr->connect();
    cout<<"##### Device connected "<<status<<"  ###  "<<endl;
    //pic
    src.create(640, 420, CV_8UC3);
    bContious = 1;
    nFrameNum = 0 ;
}

CameraDevice:: ~CameraDevice()
{
    std::cout<<"Del!!"<<std::endl;
    streamPtr = CSystem::getInstance().createStreamSource(cameraSptr);
    streamPtr->detachGrabbing(IStreamSource::Proc(&CameraDevice::onGetFrame, this));
    streamPtr->stopGrabbing();
    streamPtr.reset();
    // Stop采集图像。假设创建的流对象为 streamPtr
    streamPtr->stopGrabbing();
    TVector<ICameraPtr>vCameraPtrList;
    isDiscoverySuccess = systemObj.discovery(vCameraPtrList);
    cameraSptr = vCameraPtrList[0];
    status = cameraSptr->disConnect();
} 

int CameraDevice::init()
{
    if((status != isDiscoverySuccess ) || nDeviceNum <= 0)
    {
        cout<<"Error occured!!!!!!"<<endl;
        return 0;
    }
    // 设置相机采图模式（连续采图)
    //getGrabMode(cameraSptr, bContious);
    setGrabMode(cameraSptr, bContious);
    //set Time
    setExposureTime(cameraSptr,10000,0 );
    CSystem &systemObj = CSystem::getInstance();
    modifyCamralExposureTime(systemObj, cameraSptr);
    //set gamma
    double dExposureTime=0;
    getGamma(cameraSptr, dExposureTime);
    cout<<"gama :: read "<<dExposureTime<<endl;
    setGamma(cameraSptr, 1.5);
    getGamma(cameraSptr, dExposureTime);
    cout<<"gama :: read "<<dExposureTime<<endl;
    setGainRaw(cameraSptr, 5);
    //set rate
//    double dFrameRate = 0 ;
//    //  getAcquisitionFrameRate(cameraSptr, dFrameRate);
//     dFrameRate = 0 ;
//     setAcquisitionFrameRate(cameraSptr, dFrameRate);
    cout<<"##### paramater seted!!!  #####"<<endl;
    streamPtr = CSystem::getInstance().createStreamSource(cameraSptr);
    streamPtr->setBufferCount(1);
    streamPtr->attachGrabbing(IStreamSource::Proc(&CameraDevice::onGetFrame, this));
    std::cout<<"niuzi"<<std::endl;
    streamPtr->startGrabbing();

}
//回调函数
void CameraDevice:: onGetFrame(const CFrame& pFrame)
{
    uint64_t blockld = pFrame.getBlockId();
}
void CameraDevice :: getImage(Mat &img)
{
     if (!streamPtr->startGrabbing())
     {
        streamPtr.reset();
         //MessageBoxA(NULL, "Start Stream Grabbing Failed.", "", 0);
     }
     std::cout<<"#################### get it####################"<<std::endl;
//    if(streamPtr->attachGrabbing(){
     const void* pImage = frame.getImage();
     std::cout<<"#################### get it####################"<<std::endl;
           // 假设获取图像帧 CFrame 对象为 frame
           // 分配转换所需 buffer。注意，目标格式为 RGB、BGR 时，需要乘以 3
     int nBGRBufferSize = frame.getImageWidth() * frame.getImageHeight() * 3;
     uint8_t *pBGRbuffer = new uint8_t[nBGRBufferSize];
           // 设置转换配置参数
     IMGCNV_SOpenParam openParam;
     openParam.width = frame.getImageWidth();
     openParam.height = frame.getImageHeight();
     openParam.paddingX = frame.getImagePadddingX();
     openParam.paddingY = frame.getImagePadddingY();
     openParam.dataSize = frame.getImageSize();
     openParam.pixelForamt = frame.getImagePixelFormat();
           // 转换为 RGB24。 转换为其他格式时，调用相应的接口
     IMGCNV_EErr status = IMGCNV_ConvertToBGR24((unsigned char *)pImage,
     &openParam,
     pBGRbuffer,
     &nBGRBufferSize);
     if (IMGCNV_SUCCESS != status)
     {
        cout<<"#####  Convert failed!!!    ###"<<endl;
        delete[] pBGRbuffer; // 转码失败时，释放内存
     }

     img = cv::Mat(frame.getImageHeight(),
     frame.getImageWidth(),
     CV_8UC3,
     (uint8_t*)pBGRbuffer);
     cout<<"This is where we translate image in camera into CV::Mat"<<endl;
     nFrameNum++;
       //free(pBGRbuffer);
}

   // free(pBGRbuffer);
//    delete[] pBGRbuffer;
// void onStreamEvent(const SStreamArg& arg, void* pUser)
// {
//     if (streamEventStreamChannelError == arg.eStreamEventStatus) // 流通道错误
//     {
//     //停止拉流
//     streamPtr->stopGrabbing();
//     //重新开始拉流
//     streamPtr->startGrabbing();
//     }
//     else if (streamEventLostPacket == arg.eStreamEventStatus) // 丢包引起丢帧
//     {
//     printf("pUser:%s EventLostPacket!\n", pUser);
//     }
//     else if (streamEventNormal == arg.eStreamEventStatus)
//     {
//     printf("pUser:%s EventNormal!\n", pUser);
//     }
// }

uint64_t CameraDevice::getFrameNumber()
{
    return nFrameNum;
}
