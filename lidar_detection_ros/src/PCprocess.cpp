#include <string>

#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/PointCloud2.h>

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

#include <tf/transform_broadcaster.h>

#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_msgs/BoundingBox.h>

#include "lidar_detection_msg/Clusters.h"

namespace pcl
{
	typedef PointCloud<pcl::PointXYZI> PC_XYZI;
};

class detectDriver
{
	public:
		detectDriver();
	private:
		ros::NodeHandle node_handle;
		ros::Subscriber ousterPC2_sub;
		ros::Publisher procPoint_pub;
		ros::Publisher boundBox_pub;
		ros::Publisher clusters_pub;

		pcl::PC_XYZI::Ptr GetPCFromMsg(const sensor_msgs::PointCloud2::ConstPtr&);
		sensor_msgs::PointCloud2 GetMsgFromPC(const pcl::PC_XYZI::Ptr);
		
		pcl::PC_XYZI::Ptr PassFilter(const pcl::PC_XYZI::Ptr);
		pcl::PC_XYZI::Ptr VoxelFilter(const pcl::PC_XYZI::Ptr);
		pcl::PC_XYZI::Ptr CutGround(const pcl::PC_XYZI::Ptr, pcl::PC_XYZI::Ptr&);
		std::vector<pcl::PointIndices> GetClusters(const pcl::PC_XYZI::Ptr);
		pcl::PC_XYZI::Ptr GetPCfromIndices(const pcl::PC_XYZI::Ptr, pcl::PointIndices);

		void ousterPC2_sub_callback(const sensor_msgs::PointCloud2::ConstPtr&);
};

detectDriver::detectDriver()
{
	node_handle = ros::NodeHandle("~");
	ousterPC2_sub = node_handle.subscribe("/os1_cloud_node/points", 1, &detectDriver::ousterPC2_sub_callback, this);
	procPoint_pub = node_handle.advertise<sensor_msgs::PointCloud2>("/ProcessedPC", 10);
	boundBox_pub = node_handle.advertise<jsk_recognition_msgs::BoundingBoxArray>("/BoundingBox", 10);
	clusters_pub = node_handle.advertise<lidar_detection_msg::Clusters>("/clusters", 10);
}

pcl::PC_XYZI::Ptr detectDriver::GetPCFromMsg(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
	// PC2 -> pcl_pc2 -> pcl_pointcloid
	pcl::PCLPointCloud2 pcl_pc2;
	pcl::PC_XYZI::Ptr pointcloud(new pcl::PC_XYZI);

	pcl_conversions::toPCL(*msg, pcl_pc2);
	pcl::fromPCLPointCloud2(pcl_pc2, *pointcloud);

	return pointcloud;
}

sensor_msgs::PointCloud2 detectDriver::GetMsgFromPC(const pcl::PC_XYZI::Ptr pointcloud)
{
	// pcl_pointcloid -> pcl_pc2 -> PC2
	pcl::PCLPointCloud2 pcl_pc2;
	sensor_msgs::PointCloud2 msg;
	
	pcl::toPCLPointCloud2(*pointcloud, pcl_pc2);
	pcl_conversions::fromPCL(pcl_pc2, msg);

	return msg;
}

pcl::PC_XYZI::Ptr detectDriver::PassFilter(const pcl::PC_XYZI::Ptr pointcloud)
{
	pcl::PC_XYZI::Ptr getPoint(new pcl::PC_XYZI);
	*getPoint = *pointcloud;
	
	pcl::PassThrough<pcl::PointXYZI> ptfilter1;
	ptfilter1.setInputCloud(getPoint);
	ptfilter1.setFilterFieldName("x");
	ptfilter1.setFilterLimits(-15.0, 15.0);
	ptfilter1.setFilterLimits(-15.0, 15.0);
	ptfilter1.setNegative(false);
	ptfilter1.filter(*getPoint);

	pcl::PassThrough<pcl::PointXYZI> ptfilter2;
	ptfilter2.setInputCloud(getPoint);
	ptfilter2.setFilterFieldName("y");
	ptfilter2.setFilterLimits(-15.0, 15.0);
	ptfilter2.setFilterLimits(-15.0, 15.0);
	ptfilter2.setNegative(false);
	ptfilter2.filter(*getPoint);

	pcl::PassThrough<pcl::PointXYZI> ptfilter3;
	ptfilter3.setInputCloud(getPoint);
	ptfilter3.setFilterFieldName("z");
	ptfilter3.setFilterLimits(-15.0, 15.0);
	ptfilter3.setFilterLimits(-15.0, 15.0);
	ptfilter3.setNegative(false);
	ptfilter3.filter(*getPoint);

	return getPoint;
}

pcl::PC_XYZI::Ptr detectDriver::VoxelFilter(const pcl::PC_XYZI::Ptr pointcloud)
{
	pcl::PC_XYZI::Ptr getPoint(new pcl::PC_XYZI);
	*getPoint = *pointcloud;

	pcl::VoxelGrid<pcl::PointXYZI> vg;
	vg.setLeafSize(0.1f, 0.1f, 0.1f);		// 設置 voxel grid 小方框參數
	vg.setInputCloud(getPoint);				// 輸入 cloud
	vg.filter(*getPoint);					// 輸出到 cloud_filtered

	return getPoint;
}

pcl::PC_XYZI::Ptr detectDriver::CutGround(const pcl::PC_XYZI::Ptr pointcloud, pcl::PC_XYZI::Ptr& ground)
{
	pcl::PC_XYZI::Ptr getPoint(new pcl::PC_XYZI);
	*getPoint = *pointcloud;
	
	// 隨機採樣分割平面
	pcl::SACSegmentation<pcl::PointXYZI> seg;
	seg.setOptimizeCoefficients(true);		// 是否優化
	seg.setModelType(pcl::SACMODEL_PLANE);	// 設置模型種類
	seg.setMethodType(pcl::SAC_RANSAC);		// 設置採樣方法（RANSAC、LMedS 等）
	seg.setMaxIterations(1500);				// 設置最大迭代次數
	seg.setDistanceThreshold(0.2);			// 點到模型上的距離若超出此閥值，就表示該點不在模型上
	seg.setAxis(Eigen::Vector3f(0, 0, 1));	// 設置軸向
	seg.setEpsAngle(0.26);					// 設置軸向角度

	ground = pcl::PC_XYZI::Ptr(new pcl::PC_XYZI);
	pcl::PC_XYZI::Ptr extract_temp(new pcl::PC_XYZI);

	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);					// 採樣到的 indice
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);	// 採樣到的結果參數
	
	double nr_points = getPoint->points.size();
	for (int i = 0; i < 100; i++)
	{
		seg.setInputCloud(getPoint);
		seg.segment(*inliers, *coefficients);

		// 從 indice 提出平面點並加入地面
		pcl::ExtractIndices<pcl::PointXYZI> extract;
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

std::vector<pcl::PointIndices> detectDriver::GetClusters(const pcl::PC_XYZI::Ptr pointcloud)
{
	pcl::PC_XYZI::Ptr getPoint(new pcl::PC_XYZI);
	*getPoint = *pointcloud;
	
	pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
	tree->setInputCloud(getPoint);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;	// 歐式距離分類
	ec.setClusterTolerance(0.4); 						// 設置
	ec.setMinClusterSize(50);							// 數量下限 50
	ec.setMaxClusterSize(1000);							// 數量上限 1000
	ec.setSearchMethod(tree);							// 搜索方法，radiusSearch
	ec.setInputCloud(getPoint);
	ec.extract(cluster_indices);
	
	return cluster_indices;
}

pcl::PC_XYZI::Ptr detectDriver::GetPCfromIndices(const pcl::PC_XYZI::Ptr pointcloud, pcl::PointIndices pID)
{
	pcl::PC_XYZI::Ptr cloud_cluster(new pcl::PC_XYZI);
	pcl::ExtractIndices<pcl::PointXYZI> extract;
	extract.setInputCloud(pointcloud);
	extract.setIndices(boost::make_shared<std::vector<int>>(pID.indices));
	extract.filter(*cloud_cluster);

	return cloud_cluster;
}

void detectDriver::ousterPC2_sub_callback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
	pcl::PC_XYZI::Ptr pointcloud = GetPCFromMsg(msg);
	pcl::PC_XYZI::Ptr ground;

	pointcloud = PassFilter(pointcloud);
	pointcloud = VoxelFilter(pointcloud);
	pointcloud = CutGround(pointcloud, ground);

	std::vector<pcl::PointIndices> cluster_indices = GetClusters(pointcloud);
	std::cout << "切割出" << cluster_indices.size() << " objects" << std::endl;

	// Publish
	sensor_msgs::PointCloud2 pubPCmsg = GetMsgFromPC(pointcloud);
	pubPCmsg.header.frame_id = "my_frame";
	pubPCmsg.header.stamp = ros::Time::now();

	lidar_detection_msg::Clusters clustersMsg;
	clustersMsg.bboxArray.header.frame_id = "my_frame";
	clustersMsg.bboxArray.header.stamp = ros::Time::now();

	for (auto it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		auto cloud_cluster = GetPCfromIndices(pointcloud, *it);

		// 長寬高篩選
		pcl::PointXYZI min_pt, max_pt;
		pcl::getMinMax3D(*cloud_cluster, min_pt, max_pt);

		float length_Up = max_pt.z - min_pt.z;
		float length_Left = max_pt.y - min_pt.y;
		float length_Front = max_pt.x - min_pt.x;

		if (length_Up >= 2.8f
			|| length_Left >= 4.5f
			|| length_Front >= 6.0f
			|| abs((max_pt.x + min_pt.x) / 2) > 30.0f
			|| abs((max_pt.y + min_pt.y) / 2) > 30.0f)
			continue;

		// 建立 Bounding Box
		jsk_recognition_msgs::BoundingBox bb;
		bb.header.frame_id = "my_frame";
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

		clustersMsg.pointcloudArray.push_back(GetMsgFromPC(cloud_cluster));
		clustersMsg.bboxArray.boxes.push_back(bb);
	}

	procPoint_pub.publish(pubPCmsg);
	clusters_pub.publish(clustersMsg);
	boundBox_pub.publish(clustersMsg.bboxArray);

	// Public TF
	static tf::TransformBroadcaster br;
	tf::Transform transform;
	transform.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
	transform.setRotation(tf::Quaternion(tf::Vector3(0, 0, 1), 3.14));
	br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "my_frame"));
	br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "os1_lidar"));
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