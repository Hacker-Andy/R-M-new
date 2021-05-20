QT       += core gui printsupport
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11 console
CTEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0



# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target


INCLUDEPATH += /usr/local/include
INCLUDEPATH += /usr/local/include/opencv
INCLUDEPATH += /usr/local/include/opencv2
INCLUDEPATH += /usr/local/include/opencv2/core
INCLUDEPATH += /usr/local/include/opencv2/highgui
INCLUDEPATH += /usr/local/include/opencv2/imgproc
INCLUDEPATH += /usr/local/include/opencv2/flann
INCLUDEPATH += /usr/local/include/opencv2/photo
INCLUDEPATH += /usr/local/include/opencv2/video
INCLUDEPATH += /usr/local/include/opencv2/features2d
INCLUDEPATH += /usr/local/include/opencv2/objdetect
INCLUDEPATH += /usr/local/include/opencv2/calib3d
INCLUDEPATH += /usr/local/include/opencv2/ml
INCLUDEPATH += /usr/local/include/opencv2/contrib
#INCLUDEPATH += /usr/andy/YIHENG_ZHBIT_VISION-infantry/R-M_Robot/include
INCLUDEPATH += /opt/DahuaTech/MVviewer/include

QMAKE_LIBS_OPENGL =
DEFINES += QT_NO_DEBUG_OUTPUT LINUX64 QT_NO_OPENGL
QMAKE_CXXFLAGS_RELEASE += -mssse3
LIBS += -L./depends/	-lMVSDK
LIBS += -L/opt/DahuaTech/MVviewer/lib	-lMVSDK
#	LIBS += -L../../lib/	-llog4cpp
LIBS += -L./depends/	-lImageConvert
LIBS += -L./depends/	-lVideoRender
LIBS += -L/opt/DahuaTech/MVviewer/lib/GenICam/bin/Linux64_x64 -lGCBase_gcc421_v3_0 -lGenApi_gcc421_v3_0 -lLog_gcc421_v3_0 -llog4cpp_gcc421_v3_0 -lNodeMapData_gcc421_v3_0 -lXmlParser_gcc421_v3_0 -lMathParser_gcc421_v3_0 -lrt -lpthread


LIBS += `pkg-config opencv --cflags --libs`

#LIBS += -L /opt/DahuaTech/MVviewer/lib/Qt -IMVSDK

#LIBS += -lgxiapi \
#-ldximageproc

#INCLUDEPATH += ../dhcam_install_20181107/dh_camera/daheng-sdk-x64/sdk/genicam/library/CPP/include
#INCLUDEPATH += ../dhcam_install_20181107/dh_camera/daheng-sdk-x64/sdk/include


HEADERS += \
    armor_detection/armor_detect.h \
    buff_detection/buff_detect.h \
    camera/camera_device.h \
    common/filter/predict.h \
    common/serial/serial_port.h \
    common/solve_angle/solve_angle.h \
    common/thread_control.h \
#    include/GenICam/AcquisitionControl.h \
#    include/GenICam/AnalogControl.h \
#    include/GenICam/CAPI/IntTypes.h \
#    include/GenICam/CAPI/SDK.h \
#    include/GenICam/Camera.h \
#    include/GenICam/CameraLink/CLCamera.h \
#    include/GenICam/CameraLink/CLInterface.h \
#    include/GenICam/Defs.h \
#    include/GenICam/DeviceControl.h \
#    include/GenICam/DigitalIOControl.h \
#    include/GenICam/ErrorCode.h \
#    include/GenICam/EventSubscribe.h \
#    include/GenICam/Frame.h \
#    include/GenICam/GigE/GigECamera.h \
#    include/GenICam/GigE/GigEInterface.h \
#    include/GenICam/ISPControl.h \
#    include/GenICam/ImageFormatControl.h \
#    include/GenICam/ParameterNode.h \
#    include/GenICam/PixelType.h \
#    include/GenICam/StreamSource.h \
#    include/GenICam/System.h \
#    include/GenICam/TransportLayerControl.h \
#    include/GenICam/Usb/UsbCamera.h \
#    include/GenICam/Usb/UsbInterface.h \
#    include/GenICam/UserSetControl.h \
#    include/Infra/Allocator.h \
#    include/Infra/Assert.h \
#    include/Infra/AtomicCount.h \
#    include/Infra/CString.h \
#    include/Infra/Defs.h \
#    include/Infra/Detail/allocatorstringstorage.h \
#    include/Infra/Detail/atomic_count.hpp \
#    include/Infra/Detail/atomic_count_gcc.hpp \
#    include/Infra/Detail/atomic_count_gcc_x86.hpp \
#    include/Infra/Detail/atomic_count_pthreads.hpp \
#    include/Infra/Detail/atomic_count_solaris.hpp \
#    include/Infra/Detail/atomic_count_sync.hpp \
#    include/Infra/Detail/atomic_count_win32.hpp \
#    include/Infra/Detail/atomic_count_win64.hpp \
#    include/Infra/Detail/construct.h \
#    include/Infra/Detail/cowstringopt.h \
#    include/Infra/Detail/flex_string.h \
#    include/Infra/Detail/flex_string_details.h \
#    include/Infra/Detail/flex_string_shell.h \
#    include/Infra/Detail/ministringstorage.h \
#    include/Infra/Detail/pp.hpp \
#    include/Infra/Detail/pp_gcc.hpp \
#    include/Infra/Detail/pp_msvc.hpp \
#    include/Infra/Detail/simplestringstorage.h \
#    include/Infra/Detail/smallstringopt.h \
#    include/Infra/Detail/vectorstringstorage.h \
#    include/Infra/Error.h \
#    include/Infra/File.h \
#    include/Infra/Function.h \
#    include/Infra/FunctionTemplate.h \
#    include/Infra/Guard.h \
#    include/Infra/IntTypes.h \
#    include/Infra/Mutex.h \
#    include/Infra/PreProcessor.h \
#    include/Infra/PrintLog.h \
#    include/Infra/ReadWriteMutex.h \
#    include/Infra/RecursiveMutex.h \
#    include/Infra/Semaphore.h \
#    include/Infra/Signal.h \
#    include/Infra/SignalTemplate.h \
#    include/Infra/StaticAssert.h \
#    include/Infra/String.h \
#    include/Infra/System.h \
#    include/Infra/Thread.h \
#    include/Infra/ThreadSpecific.h \
#    include/Infra/Time.h \
#    include/Infra/Timer.h \
#    include/Infra/Types.h \
#    include/Infra/Types/IntTypes.h \
#    include/Infra/Types/Types.h \
#    include/Infra/Vector.h \
#    include/Infra/Version.h \
#    include/Media/ImageConvert.h \
#    include/Media/RecordVideo.h \
#    include/Media/VideoRender.h \
#    include/Memory/Block.h \
#    include/Memory/Buffer.h \
#    include/Memory/Defs.h \
#    include/Memory/Detail/MemPool.h \
#    include/Memory/Detail/checked_delete.hpp \
#    include/Memory/Detail/shared_count.h \
#    include/Memory/Detail/singleton.h \
#    include/Memory/Detail/singleton_pool.h \
#    include/Memory/Detail/smartptr_detail.h \
#    include/Memory/Detail/sp_counted_base.hpp \
#    include/Memory/Detail/sp_counted_base/sp_counted_base.h \
#    include/Memory/Detail/sp_counted_base/sp_counted_base_gcc_ia64.hpp \
#    include/Memory/Detail/sp_counted_base/sp_counted_base_gcc_x86.hpp \
#    include/Memory/Detail/sp_counted_base/sp_counted_base_impl.hpp \
#    include/Memory/Detail/sp_counted_base/sp_counted_base_w32.hpp \
#    include/Memory/Detail/sp_counted_impl.hpp \
#    include/Memory/EnableSharedFromThis.h \
#    include/Memory/Packet.h \
#    include/Memory/PoolAllocator.h \
#    include/Memory/RefCount.h \
#    include/Memory/ScopedPtr.h \
#    include/Memory/SharePacketHelper.h \
#    include/Memory/SharedPtr.h \
#    include/Memory/SmartObjectPtr.h \
#    include/Memory/WeakPtr.h \
    mainwindow.h \
    qcustomplot.h \
    base.h \
    camera/save_video.h

SOURCES += \
    armor_detection/armor_detect.cpp \
    buff_detection/buff_detect.cpp \
    camera/camera_device.cpp \
    common/filter/predict.cpp \
    common/serial/serial_port.cpp \
    common/solve_angle/solve_angle.cpp \
    common/main.cpp \
    common/thread_control.cpp \
    mainwindow.cpp \
    qcustomplot.cpp \
    camera/save_video.cpp

FORMS += \
    mainwindow.ui

DISTFILES += \
    camera/camera_param/cameraParam_0.xml \
    camera/camera_param/camera.xml \
    camera/camera_param/camera4mm.xml \
    camera/camera_param/camera4mm_2.xml \
    camera/camera_param/camera4mm_3.xml \
    camera/camera_param/camera4mm_5.xml \
    camera/camera_param/camera8mm.xml \
    camera/camera_param/camera8mm_1.xml \
    camera/camera_param/camera8mm_11.xml \
    camera/camera_param/galaxy_0.xml \
    camera/camera_param/galaxy_1.xml \
    camera/camera_param/galaxy_1024.xml \
    camera/camera_param/galaxy_2.xml \
    camera/camera_param/galaxy_3.xml

SUBDIRS += \
    galaxy/galaxy.pro
