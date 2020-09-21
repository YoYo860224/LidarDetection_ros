#include <string>
#include <time.h>

#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/PointCloud2.h>

// #define PCL_NO_PRECOMPILE
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/common.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/impl/passthrough.hpp>
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/common/impl/common.hpp>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>

#include <jsk_recognition_msgs/BoundingBoxArray.h>
// #include <jsk_recognition_msgs/BoundingBox.h>
#include <velodyne_pointcloud/point_types.h>

#include "lidar_detection_msg/Clusters.h"

namespace mypcl
{
    // Use XYZI.
    typedef pcl::PointXYZI PointDef;
    typedef pcl::PointCloud<pcl::PointXYZI> PC_Def;

    // Use XYZIR.
    // typedef velodyne_pointcloud::PointXYZIR PointDef;
    // typedef pcl::PointCloud<velodyne_pointcloud::PointXYZIR> PC_Def;
};

class detectDriver
{
    public:
        detectDriver();
    private:
        ros::NodeHandle node_handle;
        ros::Subscriber ousterPC2_sub;
        ros::Publisher procPoint_pub;
        ros::Publisher grndPoint_pub;
        ros::Publisher boundBox_pub;
        ros::Publisher clusters_pub;

        mypcl::PC_Def::Ptr GetPCFromMsg(const sensor_msgs::PointCloud2::ConstPtr&);
        sensor_msgs::PointCloud2 GetMsgFromPC(const mypcl::PC_Def::Ptr);

        mypcl::PC_Def::Ptr PassFilter(const mypcl::PC_Def::Ptr);
        mypcl::PC_Def::Ptr VoxelFilter(const mypcl::PC_Def::Ptr);
        mypcl::PC_Def::Ptr CutGround(const mypcl::PC_Def::Ptr, mypcl::PC_Def::Ptr&);
        std::vector<pcl::PointIndices> GetClusters(const mypcl::PC_Def::Ptr);
        mypcl::PC_Def::Ptr GetPCfromIndices(const mypcl::PC_Def::Ptr, pcl::PointIndices);

        void ousterPC2_sub_callback(const sensor_msgs::PointCloud2::ConstPtr&);
};

detectDriver::detectDriver()
{
    node_handle = ros::NodeHandle("~");
    ousterPC2_sub = node_handle.subscribe("/points_no_ground", 1, &detectDriver::ousterPC2_sub_callback, this);
    procPoint_pub = node_handle.advertise<sensor_msgs::PointCloud2>("/ProcessedPC", 10);
    grndPoint_pub = node_handle.advertise<sensor_msgs::PointCloud2>("/GroundPC", 10);
    boundBox_pub = node_handle.advertise<jsk_recognition_msgs::BoundingBoxArray>("/BoundingBox", 10);
    clusters_pub = node_handle.advertise<lidar_detection_msg::Clusters>("/clusters", 10);
}

mypcl::PC_Def::Ptr detectDriver::GetPCFromMsg(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    // PC2 -> pcl_pc2 -> pcl_pointcloid
    pcl::PCLPointCloud2 pcl_pc2;
    mypcl::PC_Def::Ptr pointcloud(new mypcl::PC_Def);

    pcl_conversions::toPCL(*msg, pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2, *pointcloud);

    return pointcloud;
}

sensor_msgs::PointCloud2 detectDriver::GetMsgFromPC(const mypcl::PC_Def::Ptr pointcloud)
{
    // pcl_pointcloid -> pcl_pc2 -> PC2
    pcl::PCLPointCloud2 pcl_pc2;
    sensor_msgs::PointCloud2 msg;

    pcl::toPCLPointCloud2(*pointcloud, pcl_pc2);
    pcl_conversions::fromPCL(pcl_pc2, msg);

    return msg;
}

mypcl::PC_Def::Ptr detectDriver::PassFilter(const mypcl::PC_Def::Ptr pointcloud)
{
    mypcl::PC_Def::Ptr getPoint(new mypcl::PC_Def);
    *getPoint = *pointcloud;

    pcl::PassThrough<mypcl::PointDef> ptfilter1;
    ptfilter1.setInputCloud(getPoint);
    ptfilter1.setFilterFieldName("x");
    ptfilter1.setFilterLimits(-35.0, 35.0);
    ptfilter1.setNegative(false);
    ptfilter1.filter(*getPoint);

    pcl::PassThrough<mypcl::PointDef> ptfilter2;
    ptfilter2.setInputCloud(getPoint);
    ptfilter2.setFilterFieldName("y");
    ptfilter2.setFilterLimits(-35.0, 35.0);
    ptfilter2.setNegative(false);
    ptfilter2.filter(*getPoint);

    pcl::PassThrough<mypcl::PointDef> ptfilter3;
    ptfilter3.setInputCloud(getPoint);
    ptfilter3.setFilterFieldName("z");
    ptfilter3.setFilterLimits(-15.0, 15.0);
    ptfilter3.setNegative(false);
    ptfilter3.filter(*getPoint);

    return getPoint;
}

mypcl::PC_Def::Ptr detectDriver::VoxelFilter(const mypcl::PC_Def::Ptr pointcloud)
{
    mypcl::PC_Def::Ptr getPoint(new mypcl::PC_Def);
    *getPoint = *pointcloud;

    pcl::VoxelGrid<mypcl::PointDef> vg;
    vg.setLeafSize(0.1f, 0.1f, 0.1f);		// 設置 voxel grid 小方框參數
    vg.setInputCloud(getPoint);				// 輸入 cloud
    vg.filter(*getPoint);					// 輸出到 cloud_filtered

    return getPoint;
}

mypcl::PC_Def::Ptr detectDriver::CutGround(const mypcl::PC_Def::Ptr pointcloud, mypcl::PC_Def::Ptr& ground)
{
    mypcl::PC_Def::Ptr getPoint(new mypcl::PC_Def);
    *getPoint = *pointcloud;

    // 隨機採樣分割平面
    pcl::SACSegmentation<mypcl::PointDef> seg;
    seg.setOptimizeCoefficients(true);                      // 是否優化
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);    // 設置模型種類
    seg.setMethodType(pcl::SAC_RANSAC);                     // 設置採樣方法（RANSAC、LMedS 等）
    seg.setMaxIterations(500);                              // 設置最大迭代次數
    seg.setDistanceThreshold(0.1);                          // 點到模型上的距離若超出此閥值，就表示該點不在模型上
    seg.setAxis(Eigen::Vector3f(0, 0, 1));                  // 設置軸向
    seg.setEpsAngle(0.1);                                   // 設置軸向角度

    ground = mypcl::PC_Def::Ptr(new mypcl::PC_Def);
    mypcl::PC_Def::Ptr extract_temp(new mypcl::PC_Def);

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);                  // 採樣到的 indice
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);   // 採樣到的結果參數

    double nr_points = getPoint->points.size();
    for (int i = 0; i < 100; i++)
    {
        seg.setInputCloud(getPoint);
        seg.segment(*inliers, *coefficients);

        // 從 indice 提出平面點並加入地面
        pcl::ExtractIndices<mypcl::PointDef> extract;
        extract.setInputCloud(getPoint);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*extract_temp);
        *ground += *extract_temp;

        // 設置反向提取，提出非平面點並設為當前點
        extract.setNegative(true);
        extract.filter(*getPoint);

        if(getPoint->points.size() < 0.9 * nr_points)
            break;
    }

    return getPoint;
}

std::vector<pcl::PointIndices> detectDriver::GetClusters(const mypcl::PC_Def::Ptr pointcloud)
{
    mypcl::PC_Def::Ptr getPoint(new mypcl::PC_Def);
    *getPoint = *pointcloud;

    pcl::search::KdTree<mypcl::PointDef>::Ptr tree(new pcl::search::KdTree<mypcl::PointDef>);
    tree->setInputCloud(getPoint);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<mypcl::PointDef> ec;    // 歐式距離分類
    ec.setClusterTolerance(0.3);                            // 設置
    ec.setMinClusterSize(30);                               // 數量下限 50
    ec.setMaxClusterSize(1000);                             // 數量上限 1000
    ec.setSearchMethod(tree);                               // 搜索方法，radiusSearch
    ec.setInputCloud(getPoint);
    ec.extract(cluster_indices);

    return cluster_indices;
}

mypcl::PC_Def::Ptr detectDriver::GetPCfromIndices(const mypcl::PC_Def::Ptr pointcloud, pcl::PointIndices pID)
{
    mypcl::PC_Def::Ptr cloud_cluster(new mypcl::PC_Def);
    pcl::ExtractIndices<mypcl::PointDef> extract;
    extract.setInputCloud(pointcloud);
    extract.setIndices(boost::make_shared<std::vector<int>>(pID.indices));
    extract.filter(*cloud_cluster);

    return cloud_cluster;
}

int frID = 0;
int myID = 1;

void detectDriver::ousterPC2_sub_callback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    mypcl::PC_Def::Ptr pointcloud = GetPCFromMsg(msg);
    // mypcl::PC_Def::Ptr ground;
    std::cout << "=======================" << std::endl;
    std::cout << "PC size:" << pointcloud->size() << std::endl;
    double t1 = clock();
    pointcloud = PassFilter(pointcloud);
    pointcloud = VoxelFilter(pointcloud);
    // pointcloud = CutGround(pointcloud, ground);

    std::vector<pcl::PointIndices> cluster_indices = GetClusters(pointcloud);
    double t5 = clock();

    std::cout << "PC size:" << pointcloud->size() << std::endl;
    std::cout << "切割出" << cluster_indices.size() << " objects" << std::endl;
    std::cout << "FPS " << 1.0 / ((t5 - t1) / CLOCKS_PER_SEC) << std::endl;

    // Publish
    sensor_msgs::PointCloud2 pubPCmsg = GetMsgFromPC(pointcloud);
    pubPCmsg.header.frame_id = "velodyne";
    pubPCmsg.header.stamp = ros::Time::now();

    // sensor_msgs::PointCloud2 pubGDmsg = GetMsgFromPC(ground);
    // pubGDmsg.header.frame_id = "velodyne";
    // pubGDmsg.header.stamp = ros::Time::now();

    // Cluster
    lidar_detection_msg::Clusters clustersMsg;
    clustersMsg.bboxArray.header.frame_id = "velodyne";
    clustersMsg.bboxArray.header.stamp = ros::Time::now();

    jsk_recognition_msgs::BoundingBoxArray bba;
    bba.header.frame_id = "velodyne";
    bba.header.stamp = ros::Time::now();

    pcl::io::savePCDFileBinary(((std::string)"/home/yoyo/桌面/hd32pcd/" + std::to_string(frID) + "_0.pcd"), *pointcloud);
    frID++;
    myID = 1;
    for (auto it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
        auto cloud_cluster = GetPCfromIndices(pointcloud, *it);

        // 長寬高篩選
        mypcl::PointDef min_pt, max_pt;
        pcl::getMinMax3D(*cloud_cluster, min_pt, max_pt);

        float length_Up = max_pt.z - min_pt.z;
        float length_Left = max_pt.y - min_pt.y;
        float length_Front = max_pt.x - min_pt.x;

        if (length_Up >= 3.0f || length_Up <= 0.5f || max_pt.z > 3.0f || max_pt.z < -3.0f)
            continue;

        // 建立 Bounding Box
        jsk_recognition_msgs::BoundingBox bb;
        bb.header.frame_id = "velodyne";
        bb.header.stamp = ros::Time::now();

        bb.pose.position.x = (max_pt.x + min_pt.x) / 2.0;
        bb.pose.position.y = (max_pt.y + min_pt.y) / 2.0;
        bb.pose.position.z = (max_pt.z + min_pt.z) / 2.0;

        bb.pose.orientation.x = 0;
        bb.pose.orientation.y = 0;
        bb.pose.orientation.z = 0;
        bb.pose.orientation.w = 1;

        bb.dimensions.x = (max_pt.x - min_pt.x);
        bb.dimensions.y = (max_pt.y - min_pt.y);
        bb.dimensions.z = (max_pt.z - min_pt.z);

        pcl::io::savePCDFileBinary(((std::string)"/home/yoyo/桌面/hd32pcd/" + std::to_string(frID) + "_" + std::to_string(myID) + ".pcd"), *cloud_cluster);
        myID++;

        clustersMsg.pointcloudArray.push_back(GetMsgFromPC(cloud_cluster));
        clustersMsg.bboxArray.boxes.push_back(bb);
        bba.boxes.push_back(bb);
    }

    procPoint_pub.publish(pubPCmsg);
    // grndPoint_pub.publish(pubGDmsg);
    clusters_pub.publish(clustersMsg);
    std::cout << bba.boxes.size() << std::endl;
    boundBox_pub.publish(bba);
}

int main( int argc, char** argv)
{
    ros::init(argc, argv, "PCprocess");
    detectDriver node;
    ros::Rate r(60);

    while (ros::ok())
    {
        ros::spinOnce();
        r.sleep();
    }

    return 0;
}