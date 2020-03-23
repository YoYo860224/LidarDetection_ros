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

#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_msgs/BoundingBox.h>

#include <tf/transform_broadcaster.h>


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

		pcl::PC_XYZI::Ptr GetPCFromMsg(const sensor_msgs::PointCloud2::ConstPtr&);
		sensor_msgs::PointCloud2 GetMsgFromPC(const pcl::PC_XYZI::Ptr);
		
		pcl::PC_XYZI::Ptr PassFilter(pcl::PC_XYZI::Ptr);
		pcl::PC_XYZI::Ptr VoxelFilter(pcl::PC_XYZI::Ptr);
		pcl::PC_XYZI::Ptr CutGround(pcl::PC_XYZI::Ptr, pcl::PC_XYZI::Ptr&);
		
		void ousterPC2_sub_callback(const sensor_msgs::PointCloud2::ConstPtr&);
};

detectDriver::detectDriver()
{
	node_handle = ros::NodeHandle("~");
	ousterPC2_sub = node_handle.subscribe("/os1_cloud_node/points", 1, &detectDriver::ousterPC2_sub_callback, this);
	procPoint_pub = node_handle.advertise<sensor_msgs::PointCloud2>("/ProcPoint", 1);
	boundBox_pub = node_handle.advertise<jsk_recognition_msgs::BoundingBoxArray>("/BoubdingBox", 1);
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
	vg.setInputCloud(getPoint);			// 輸入		cloud
	vg.filter(*getPoint);					// 輸出到	cloud_filtered

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


void detectDriver::ousterPC2_sub_callback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
	pcl::PC_XYZI::Ptr pointcloud = GetPCFromMsg(msg);
	pcl::PC_XYZI::Ptr ground;

	pointcloud = PassFilter(pointcloud);
	pointcloud = VoxelFilter(pointcloud);
	pointcloud = CutGround(pointcloud, ground);

	
	pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
	tree->setInputCloud(pointcloud);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;	// 欧式距离分类
	ec.setClusterTolerance(0.4); 						// 2cm设置ClusterTolerance，该值为近临搜索半径 0.2
	ec.setMinClusterSize(50);							// 數量下限 50
	ec.setMaxClusterSize(1000);							// 數量上限 1000
	ec.setSearchMethod(tree);							// 搜索方法，radiusSearch
	ec.setInputCloud(pointcloud);
	ec.extract(cluster_indices);
	std::cout << "切割出" << cluster_indices.size() << " objects" << std::endl;

	// Publish
	sensor_msgs::PointCloud2 pubPCmsg = GetMsgFromPC(pointcloud);
	pubPCmsg.header.frame_id = "my_frame";
	pubPCmsg.header.stamp = ros::Time::now();

	procPoint_pub.publish(pubPCmsg);

	jsk_recognition_msgs::BoundingBoxArray bbArr;
	bbArr.header.frame_id = "my_frame";
	bbArr.header.stamp = ros::Time::now();

	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		pcl::PC_XYZI::Ptr cloud_cluster(new pcl::PC_XYZI);
		pcl::ExtractIndices<pcl::PointXYZI> extract;
		extract.setInputCloud(pointcloud);
		extract.setIndices(boost::make_shared<std::vector<int>>(it->indices));
		extract.filter (*cloud_cluster);
		
		pcl::PointXYZI min_pt, max_pt;
		pcl::getMinMax3D(*cloud_cluster, min_pt, max_pt);

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

		bbArr.boxes.push_back(bb);
	}
	boundBox_pub.publish(bbArr);

	static tf::TransformBroadcaster br;
	tf::Transform transform;
	transform.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
	transform.setRotation(tf::Quaternion(tf::Vector3(0, 0, 1), 3.14));
	br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "my_frame"));
	br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "os1_lidar"));

	return;
}

int main( int argc, char** argv)
{
	ros::init(argc, argv, "detect");
	detectDriver node;
	ros::Rate r(60);

	while (ros::ok())
	{
		ros::spinOnce();
		r.sleep();
	}

	return 0;
}