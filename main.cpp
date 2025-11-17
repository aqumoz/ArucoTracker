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

struct Line {
    cv::Point2f p1, p2;
};

struct VideoAnalysis {
    std::string video_path;
    Line line;
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

static bool click1 = true; // is first click
void draw_line_callback(int event, int x, int y, int, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN)
        return;

    auto* line = reinterpret_cast<Line*>(userdata);
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
    cv::namedWindow("Draw Line");
    cv::setMouseCallback("Draw Line", draw_line_callback, &line);

    std::cout << "Draw a line by clicking two points on the first frame of: " << video_path << std::endl;
    cv::Mat temp = frame.clone();
    cv::imshow("Draw Line", temp);
    do {
        // std::cout << "test " << line.p1.x << ' ' << line.p1.y << std::endl;
        if (line.p1 != cv::Point2f()){
            cv::circle(temp, line.p1, 5, cv::Scalar(0,255,0), -1);
            cv::imshow("Draw Line", temp);
        }
        if (cv::waitKey(30) == 27) break; // ESC to exit

    } while (line.p1 == cv::Point2f() || line.p2 == cv::Point2f());

    cv::circle(temp, line.p2, 5, cv::Scalar(0,255,0), -1);

    Line extended_line = extend_line_to_frame(line, cv::Size(frame.cols, frame.rows));
    cv::line(temp, extended_line.p1, extended_line.p2, cv::Scalar(255,0,0), 2);

    // cv::line(temp, line.p1, line.p2, cv::Scalar(0,0,255), 2);

    cv::imshow("Draw Line", temp);

    while(1){
        int key = cv::waitKey(30);
        if(key == '-')
            break;
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

void analyze_video(const VideoAnalysis& analysis, bool overlay = false) {
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
    csv << "\"frame\",\"x\",\"y\",\"distance_to_line\"\n";

    // Setup output video if overlay is enabled
    cv::VideoWriter video_out;
    if (overlay) {
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        cv::Size frame_size(cap.get(cv::CAP_PROP_FRAME_WIDTH), 
                           cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);
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
        if (obj_pos.x >= 0 && obj_pos.y >= 0)
            dist = point_line_distance(obj_pos, analysis.line);
        
        if (overlay) {
            // Calculate extended line points
            cv::Size frame_size(frame.cols, frame.rows);
            Line extended = extend_line_to_frame(analysis.line, frame_size);
            
            // Draw extended line in magenta
            cv::line(frame, extended.p1, extended.p2, cv::Scalar(255, 0, 255), 2);
            
            // Draw green circles at original line endpoints
            cv::circle(frame, analysis.line.p1, 5, cv::Scalar(0, 255, 0), -1);
            cv::circle(frame, analysis.line.p2, 5, cv::Scalar(0, 255, 0), -1);
            
            // Draw cross at tracked object position if found
            if (obj_pos.x >= 0 && obj_pos.y >= 0) {
                int cross_size = 10;
                cv::line(frame, 
                        cv::Point(obj_pos.x - cross_size, obj_pos.y - cross_size),
                        cv::Point(obj_pos.x + cross_size, obj_pos.y + cross_size),
                        cv::Scalar(0, 0, 255), 2);
                cv::line(frame,
                        cv::Point(obj_pos.x - cross_size, obj_pos.y + cross_size),
                        cv::Point(obj_pos.x + cross_size, obj_pos.y - cross_size),
                        cv::Scalar(0, 0, 255), 2);
            }
            
            video_out.write(frame);
        }
        
        csv << frame_num << "," << obj_pos.x << "," << obj_pos.y << "," << dist << "\n";
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
    
    program.add_description("Analyze movement in videos relative to a reference line");
    
    // Add arguments
    program.add_argument("input")
        .help("Folder containing video files to analyze")
        .nargs(0, 1);

    program.add_argument("--gen-aruco")
        .help("Generate ArUco markers: --gen-aruco <number> <output_path>")
        .nargs(2)
        .default_value(std::vector<std::string>());
        
    program.add_argument("--overlay")
        .help("Create video with analysis visualization overlay")
        .default_value(false)
        .implicit_value(true);
        
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
    bool overlay = program.get<bool>("--overlay");
    
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

    int video_count = 0;
    std::cout << "Analysing videos" << std::endl;
    for (const auto& analysis : analyses) {
        analyze_video(analysis, overlay);
        std::cout << "Video: " << ++video_count << std::endl;
    }

    return 0;
}