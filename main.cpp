#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/aruco.hpp>
#include "include/argparse.hpp"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

namespace fs = std::filesystem;

bool measure = false;
bool overlay = false;
double px_to_mm = 1.0;

struct Line {
    cv::Point2f p1, p2;
    bool operator==(const Line& other) const {
        return p1 == other.p1 && p2 == other.p2;
    }
};

struct VideoAnalysis {
    std::string video_path;
    Line line;
    cv::Point2f start;
    float total_dist;
};

std::vector<std::string> get_video_files(const std::string& folder) {
    std::vector<std::string> videos;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv") {
                videos.push_back(entry.path().string());
            }
        }
    }
    return videos;
}

Line extend_line_to_frame(const Line& line, const cv::Size& frame_size) {
    Line ll;
    float w = frame_size.width;
    float h = frame_size.height;

    float dx = line.p2.x - line.p1.x;
    float dy = line.p2.y - line.p1.y;
    
    // P1
    ll.p1.y = line.p1.y + (-line.p1.x / dx) * dy;
    if(ll.p1.y < 0){
        ll.p1.x = line.p1.x + (-line.p1.y / dy) * dx;
        ll.p1.y = 0;
    }
    else if(ll.p1.y > h){
        ll.p1.x = line.p1.x + ((h - line.p1.y) / dy) * dx;
        ll.p1.y = h;
    }
    else
        ll.p1.x = 0;

    // P2
    ll.p2.y = line.p2.y + ((w - line.p2.x) / dx) * dy;
    if(ll.p2.y < 0){
        ll.p2.x = line.p2.x + (-line.p2.y / dy) * dx;
        ll.p2.y = 0;
    }
    else if(ll.p2.y > h){
        ll.p2.x = line.p2.x + ((h - line.p2.y) / dy) * dx;
        ll.p2.y = h;
    }
    else
        ll.p2.x = w;

    return ll;
}

float dist(const cv::Point2f& p1, const cv::Point2f& p2){
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return std::sqrt(dx*dx + dy*dy);
}

static bool click1 = true; // is first click
void draw_line_callback(int event, int x, int y, int, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN)
        return;

    Line* line = reinterpret_cast<Line*>(userdata);
    // line[1] = line[0]; // previous line = line;
    if(click1)
        line->p1 = cv::Point2f(x, y);
    else
        line->p2 = cv::Point2f(x, y);
    click1 = !click1;
}

Line get_line_from_user(const std::string& video_path) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << video_path << std::endl;
        return {};
    }
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
        std::cerr << "Failed to read first frame: " << video_path << std::endl;
        return {};
    }

    Line line;
    Line prev_line;
    cv::namedWindow("Draw Line");
    cv::setMouseCallback("Draw Line", draw_line_callback, &line);

    std::cout << "Draw a line by clicking two points on the first frame of: " << video_path << std::endl;
    cv::Mat temp;
    cv::imshow("Draw Line", frame);
    int key;
    bool first_point = true; // Is used to track, if it is the first point or second point getting placed.

    while (cv::waitKey(30) != '-'){// Hyphen to continue

        if(line == prev_line)
            continue;
        else
            prev_line = line;

        if (first_point){
            temp = frame.clone();
            cv::circle(temp, line.p1, 5, cv::Scalar(0, 255, 0), -1);
            cv::imshow("Draw Line", temp);
            first_point = false;
        }
        else {
            if(measure){
                cv::line(temp, line.p1, line.p2, cv::Scalar(255,0,0), 2);
                std::cout << "Distance [px]: " << dist(line.p1, line.p2) << std::endl;
            }
            else{
                Line extended_line = extend_line_to_frame(line, cv::Size(frame.cols, frame.rows));
                cv::line(temp, extended_line.p1, extended_line.p2, cv::Scalar(255,0,0), 2);
            }
            cv::circle(temp, line.p1, 5, cv::Scalar(0, 255, 0), -1);
            cv::circle(temp, line.p2, 5, cv::Scalar(0,0,255), -1);
            cv::imshow("Draw Line", temp);
            first_point = true;
        }
    }

    cv::destroyWindow("Draw Line");
    return line;
}


float point_line_distance(const cv::Point2f& pt, const Line& line) {
    cv::Point2f v = line.p2 - line.p1;
    cv::Point2f w = pt - line.p1;
    float c1 = v.dot(w);
    float c2 = v.dot(v);
    float b = c1 / c2;
    cv::Point2f pb = line.p1 + b * v;
    return cv::norm(pt - pb);
}


cv::Point2f project_marker_onto_line(const cv::Point2f& pt, const Line& line){
    cv::Point2f pt_ = pt - line.p1;
    cv::Point2f p2 = line.p2 - line.p1;

    return (pt_.dot(p2) / p2.dot(p2)) * p2 + line.p1;
}


float dist_from_start(uint frame_num, const cv::Point2f& pt, VideoAnalysis& analysis) {
    const cv::Point2f& end = analysis.line.p2;
    if(frame_num == 0){
        analysis.start = project_marker_onto_line(pt, analysis.line);
        analysis.total_dist = dist(analysis.start, end);
    }

    const cv::Point2f marker_ = project_marker_onto_line(pt, analysis.line);

    return dist(analysis.start, marker_);
}


cv::Point2f track_aruco(const cv::Mat& frame) {
    std::vector<std::vector<cv::Point2f>> markerCorners;
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> rejectedCandidates;
    cv::Ptr<cv::aruco::DetectorParameters> detectorParams = cv::aruco::DetectorParameters::create();
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, detectorParams, rejectedCandidates);
    
    // Return center of first detected marker, or (-1,-1) if none found
    if (!markerCorners.empty()) {
        cv::Point2f center(0, 0);
        for (const auto& corner : markerCorners[0]) {
            center += corner;
        }
        return center * 0.25f;
    }
    return cv::Point2f(-1, -1);
}

cv::Point2f track_red_object(const cv::Mat& frame) {
    cv::Mat hsv, mask;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    // Red color range in HSV (may need tuning)
    cv::Scalar lower1(0, 120, 70), upper1(10, 255, 255);
    cv::Scalar lower2(170, 120, 70), upper2(180, 255, 255);
    cv::inRange(hsv, lower1, upper1, mask);
    cv::Mat mask2;
    cv::inRange(hsv, lower2, upper2, mask2);
    mask |= mask2;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double max_area = 0;
    cv::Point2f center(-1, -1);
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > max_area) {
            cv::Moments m = cv::moments(contour);
            if (m.m00 != 0) {
                center = cv::Point2f(m.m10 / m.m00, m.m01 / m.m00);
                max_area = area;
            }
        }
    }
    return center;
}

void applyGamma(cv::Mat& img, double gamma)
{
    CV_Assert(img.type() == CV_8UC1);

    // 1. Precompute lookup table
    cv::Mat lut(1, 256, CV_8U);
    uchar* p = lut.ptr<uchar>();
    for (int i = 0; i < 256; ++i)
    {
        p[i] = cv::saturate_cast<uchar>(std::pow(i / 255.0, gamma) * 255.0);
    }

    // 2. Apply LUT (OpenCV handles SIMD + multithreading internally)
    cv::LUT(img, lut, img);
}


/**
 * Analyzes the video.
 *
 * @param analysis   The analysis settings.
 * @param overlay    Whether to draw the overlay on the frames.
 * @param px_to_mm   Conversion factor from pixels to millimeters. (default is 1 (no conversion))
 */
void analyze_video(VideoAnalysis& analysis) {
    cv::VideoCapture cap(analysis.video_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << analysis.video_path << std::endl;
        return;
    }
    
    
    std::string outPath = analysis.video_path;
    // remove file type hint from out path.
    int dotIdx = outPath.find_last_of('.');
    if(dotIdx > 0)
        outPath = outPath.substr(0, dotIdx);
    
    if(!std::filesystem::exists(outPath))
        std::filesystem::create_directory(outPath);

    std::string csv_path = outPath + "/analysis.csv";
    std::ofstream csv(csv_path);
    csv << "\"frame\",\"x\",\"y\",\"distance_to_line_px\",\"distance_to_line_mm\",\"distance_from_start_in_px\",\"distance_from_start_in_mm\",\"time_in_seconds\"\n";

    // Setup output video if overlay is enabled
    cv::VideoWriter video_out;
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (overlay) {
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        cv::Size frame_size(cap.get(cv::CAP_PROP_FRAME_WIDTH), 
                           cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        std::string overlay_path = outPath + "/overlay.mp4";
        video_out.open(overlay_path, fourcc, fps, frame_size, true);
        
        if (!video_out.isOpened()) {
            std::cerr << "Failed to create output video: " << overlay_path << std::endl;
            return;
        }
    }

    int frame_num = 0;
    cv::Mat tmp;
    cv::Mat frame;
    float gamma = 2.0;

    while (cap.read(frame)) {
        // cv::Point2f obj_pos = track_red_object(frame);
        // cv::cvtColor(tmp, frame, cv::COLOR_BGR2GRAY, 1);
        // applyGamma(frame, 2.5);
        // frame.convertTo(frame, -1, 0.5, 0);

        cv::Point2f obj_pos = track_aruco(frame);
        float dist = -1;
        float dist_from_start_px = -1.0;
        float dist_from_start_mm = -1.0;
        if (obj_pos.x >= 0 && obj_pos.y >= 0)
            dist = point_line_distance(obj_pos, analysis.line);

            // The distance from the start to end, projected along the line.
            dist_from_start_px = dist_from_start(frame_num, obj_pos, analysis);

            // If the marker have moved beyond the end point, then stop.
            if(dist_from_start_px > analysis.total_dist)
                break;

            dist_from_start_mm = dist_from_start_px * px_to_mm;
        
        if (overlay) {
            // Calculate extended line points
            cv::Size frame_size(frame.cols, frame.rows);
            Line extended = extend_line_to_frame(analysis.line, frame_size);
            
            // Draw extended line in magenta
            cv::line(frame, extended.p1, extended.p2, cv::Scalar(255, 0, 255), 2);
            
            if (obj_pos.x >= 0 && obj_pos.y >= 0) {
                int cross_size = 10;
                // Draw cross at tracked object position if found
                cv::line(frame, 
                        cv::Point(obj_pos.x - cross_size, obj_pos.y - cross_size),
                        cv::Point(obj_pos.x + cross_size, obj_pos.y + cross_size),
                        cv::Scalar(0, 0, 255), 2);

                cv::line(frame,
                        cv::Point(obj_pos.x - cross_size, obj_pos.y + cross_size),
                        cv::Point(obj_pos.x + cross_size, obj_pos.y - cross_size),
                        cv::Scalar(0, 0, 255), 2);

                // Draw line from marker to line.
                cv::line(frame,
                        cv::Point(obj_pos.x, obj_pos.y),
                        project_marker_onto_line(obj_pos, analysis.line),
                        cv::Scalar(0, 255, 0), 2);
            }

            // Draw circles at original line endpoints
            cv::circle(frame, analysis.line.p1, 5, cv::Scalar(127, 127, 0), -1);
            cv::circle(frame, analysis.line.p2, 5, cv::Scalar(0, 255, 0), -1);

            // Draw start point
            cv::circle(frame, analysis.start, 5, cv::Scalar(0, 255, 255), -1);
            
            video_out.write(frame);
        }
        float time_in_seconds = frame_num/fps;
        csv << frame_num << "," << obj_pos.x << "," << obj_pos.y << "," << dist << "," << dist * px_to_mm << "," << dist_from_start_px << "," << dist_from_start_mm << "," << time_in_seconds << std::endl;
        frame_num++;
    }
    
    csv.close();
    if (overlay) {
        video_out.release();
    }
    std::cout << "Analysis complete for: " << analysis.video_path << ", results saved to: " << csv_path << std::endl;
}

void generate_aruco(int num_markers, std::string output_path){
    if(!std::filesystem::exists(output_path)) {
        if(!std::filesystem::create_directories(output_path)) {
            std::cerr << "Failed to create output directory: " << output_path << std::endl;
            return;
        }
    }
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    for (int i = 0; i < num_markers; i++) {
        cv::Mat markerImage;
        cv::aruco::drawMarker(dictionary, i, 200, markerImage, 1);
        cv::imwrite(output_path + "/marker" + std::to_string(i) + ".png", markerImage);
    }
    std::cout << "Generated " << num_markers << " ArUco markers in " << output_path << std::endl;
}

int main(int argc, char** argv) {
    argparse::ArgumentParser program("video_analyzer");
    
    program.add_description("Analyze movement of an aruco marker in videos\nrelative to a reference line and a start and end point.");
    
    // Add arguments
    program.add_argument("input")
        .help("Folder containing video files to analyze.")
        .nargs(0, 1);
    
    program.add_argument("-c", "--px-mm")
        .help("The number used to convert from pixels to millimeters.")
        .store_into(px_to_mm);

    program.add_argument("-g", "--gen-aruco")
        .help("Generate ArUco markers: --gen-aruco <number> <output_path>.")
        .nargs(2)
        .default_value(std::vector<std::string>());

    program.add_argument("-m", "--measure-px")
        .help("Shows the first frame and prints the distance between two points you click on.")
        .store_into(measure)
        .flag();
        
    program.add_argument("-ol", "--overlay")
        .help("Create video with analysis visualization overlay.")
        .store_into(overlay);
        
    // Parse arguments
    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }
    
    // Get values
    auto gen_aruco = program.get<std::vector<std::string>>("--gen-aruco");
    if (!gen_aruco.empty()) {
        try{
            generate_aruco(std::stoi(gen_aruco[0]), gen_aruco[1]);
        }
        catch (const std::invalid_argument& e) {
            std::cerr << "Error: First argument to --gen-aruco must be a number" << std::endl;
            return 1;
        }
    }

    std::string folder = program.get<std::string>("input");

    std::vector<std::string> videos = get_video_files(folder);
    if (videos.empty()) {
        std::cout << "No video files found in folder: " << folder << std::endl;
        return 1;
    }

    std::vector<VideoAnalysis> analyses;
    for (const auto& video : videos) {
        Line line = get_line_from_user(video);
        analyses.push_back({video, line});
    }

    if(measure)
        return 0;

    int video_count = 0;
    std::cout << "Analysing videos" << std::endl;
    for (auto& analysis : analyses) {
        analyze_video(analysis);
        std::cout << "Video: " << ++video_count << std::endl;
    }

    return 0;
}