#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/astar_search.hpp>
#include <cmath>
#include <vector>
#include <algorithm>

// Custom vertex properties for the Boost Graph
struct VertexProperties {
  int x, y;
  bool visited;
};

// Custom edge properties for the Boost Graph
typedef boost::property<boost::edge_weight_t, double> EdgeProperties;

// Define graph type
typedef boost::adjacency_list<boost::listS, boost::vecS, boost::undirectedS, VertexProperties, EdgeProperties> Graph;

template <class Graph, class CostType>
class euclidean_distance_heuristic : public boost::astar_heuristic<Graph, CostType> {
public:
  euclidean_distance_heuristic(const Graph& g, int goal_x, int goal_y)
    : m_g(g), m_goal_x(goal_x), m_goal_y(goal_y) {}

  CostType operator()(typename boost::graph_traits<Graph>::vertex_descriptor v) {
    // Ensure that the vertex descriptor is valid.
    if (v < 0 || v >= boost::num_vertices(m_g)) {
      throw std::runtime_error("euclidean_distance_heuristic: Invalid vertex descriptor.");
    }

    // Calculate the Euclidean distance from the current vertex to the goal.
    double dx = m_g[v].x - m_goal_x;
    double dy = m_g[v].y - m_goal_y;
    return static_cast<CostType>(std::sqrt(dx * dx + dy * dy));
  }

private:
  const Graph& m_g;
  int m_goal_x, m_goal_y;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
};

// Custom visitor for A* search
template <typename Vertex>
class astar_goal_visitor : public boost::default_astar_visitor {
public:
  astar_goal_visitor(Vertex goal) : m_goal(goal) {}

  template <typename Graph>
  void examine_vertex(Vertex u, const Graph& g) {
    if (u == m_goal) {
      throw std::runtime_error("Found the goal");
    }
  }

private:
  Vertex m_goal;
};

class GlobalPlanner : public rclcpp::Node, tf_listener_(tf_buffer_)
{
public:
  GlobalPlanner() : Node("global_planner") {
    map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>("/map", 10, std::bind(&GlobalPlanner::map_callback, this, std::placeholders::_1));
    goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>("/goal_pose", 10, std::bind(&GlobalPlanner::goal_callback, this, std::placeholders::_1));
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/path", 10);
  }

private:
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  nav_msgs::msg::OccupancyGrid::SharedPtr map_;
  Graph g_;

  void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    map_ = msg;
    // Clear the graph and add vertices
    g_.clear();
    for (int y = 0; y < msg->info.height; ++y) {
      for (int x = 0; x < msg->info.width; ++x) {
        boost::add_vertex(VertexProperties{x, y, false}, g_);
      }
    }
  int width = msg->info.width;
  int height = msg->info.height;
  double resolution = msg->info.resolution;

  // Output map information
  RCLCPP_INFO(this->get_logger(), "Map subscribed: width=%d, height=%d, resolution=%.2f", width, height, resolution);

    // Add edges with cost (weight)
    for (int y = 0; y < msg->info.height; ++y) {
      for (int x = 0; x < msg->info.width; ++x) {
        int current = y * msg->info.width + x;
        if (msg->data[current] == 0) { // Only add edges for free space
          if (x > 0) {
            int left = y * msg->info.width + x - 1;
            if (msg->data[left] == 0) {
              double weight = 1.0;
              boost::add_edge(current, left, EdgeProperties{weight}, g_);
            }
          }
          if (y > 0) {
            int up = (y - 1) * msg->info.width + x;
            if (msg->data[up] == 0) {
              double weight = 1.0;
              boost::add_edge(current, up, EdgeProperties{weight}, g_);
            }
          }
          if (x < msg->info.width - 1) {
            int right = y * msg->info.width + x + 1;
            if (msg->data[right] == 0) {
              double weight = 1.0;
              boost::add_edge(current, right, EdgeProperties{weight}, g_);
            }
          }
          if (y < msg->info.height - 1) {
            int down = (y + 1) * msg->info.width + x;
            if (msg->data[down] == 0) {
              double weight = 1.0;
              boost::add_edge(current, down, EdgeProperties{weight}, g_);
            }
          }
        }
      }
    } 
    RCLCPP_INFO(this->get_logger(), "Graph created with %u vertices and %u edges", boost::num_vertices(g_), boost::num_edges(g_));
  }

  void goal_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    if (!map_) {
      RCLCPP_WARN(this->get_logger(), "Map not received yet, cannot plan");
      return;
    }

    // Find start and goal vertices in the graph
int start_x = static_cast<int>((msg->pose.position.x - map_->info.origin.position.x) / map_->info.resolution);
int start_y = static_cast<int>((msg->pose.position.y - map_->info.origin.position.y) / map_->info.resolution);
int goal_x = static_cast<int>((msg.pose.position.x - map_->info.origin.position.x) / map_->info.resolution);
int goal_y = static_cast<int>((msg.pose.position.y - map_->info.origin.position.y) / map_->info.resolution);

if (start_x < 0 || start_x >= map_->info.width || start_y < 0 || start_y >= map_->info.height ||
    goal_x < 0 || goal_x >= map_->info.width || goal_y < 0 || goal_y >= map_->info.height) {
  RCLCPP_ERROR(this->get_logger(), "Start or goal position is out of map bounds.");
  return;
}
    int start_index = start_y * map_->info.width + start_x;
    int goal_index = goal_y * map_->info.width + goal_x;
    RCLCPP_INFO(this->get_logger(), "Start position: (%d, %d), index: %d", start_x, start_y, start_index);
    RCLCPP_INFO(this->get_logger(), "Goal position: (%d, %d), index: %d", goal_x, goal_y, goal_index);
    RCLCPP_INFO(this->get_logger(), "Received goal data");
    // A* search
    std::vector<double> distances(boost::num_vertices(g_), std::numeric_limits<double>::max());
    distances[start_index] = 0.0;
    std::vector<boost::graph_traits<Graph>::vertex_descriptor> predecessors(boost::num_vertices(g_));
    try {
      boost::astar_search(
        g_, start_index,
        euclidean_distance_heuristic<Graph, double>(g_, goal_x, goal_y),
        boost::predecessor_map(&predecessors[0]).distance_map(&distances[0]).visitor(astar_goal_visitor(goal_index))
      );
    }
    catch (const std::runtime_error& e) {
      if (std::string(e.what()) != "Found the goal") {
        RCLCPP_ERROR(this->get_logger(), "A* search failed: %s", e.what());
        return;
      }
    }

    // Reconstruct the path
    nav_msgs::msg::Path path;
    path.header.frame_id = map_->header.frame_id;
    path.header.stamp = this->now();
    for (int v = goal_index; v != start_index; v = predecessors[v]) {
      if (predecessors[v] == v) {
        RCLCPP_ERROR(this->get_logger(), "Path reconstruction failed, no path found");
        return;
      }

      geometry_msgs::msg::PoseStamped pose;
      pose.header = path.header;
      pose.pose.position.x = map_->info.origin.position.x + g_[v].x * map_->info.resolution;
      pose.pose.position.y = map_->info.origin.position.y + g_[v].y * map_->info.resolution;
      pose.pose.orientation.w = 1.0;
      path.poses.push_back(pose);
    }
    std::reverse(path.poses.begin(), path.poses.end());
    path_pub_->publish(path);
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GlobalPlanner>());
  rclcpp::shutdown();
  return 0;
}
