#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <string>
#include <netinet/in.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include <mariadb/mysql.h>
#include <fstream>
#include <map>
#include <arpa/inet.h>
#include <random>
#include <ctime>
#include <iomanip>
#include <sstream>
#include "base64.h"
#include <iterator>
#include <algorithm>
#include <stdexcept>
#include <openssl/evp.h>  // OpenSSL 라이브러리 헤더 (Base64 인코딩을 위해 사용)
#include <openssl/bio.h>  // BIO와 관련된 기능을 사용하기 위해 추가
#include <openssl/ssl.h>  // SSL 관련 작업을 위한 헤더
#include <chrono>

using json = nlohmann::json;
using namespace std;

string getCurrentTimeString();
int get_socket_for_ip(const std::string& ip);
string getCurrentTimestamp();
string get_first_utf8_char(const std::string& str);
string getCurrentDateOnly();

std::mutex cout_mutex;
std::string buffer;
std::map<std::string, int> ip_to_socket;
std::mutex ip_map_mutex;

constexpr int PORT = 10008;
constexpr int BUFFER_SIZE = 100000;


void send_json(int client_sock, const json& response) {
    std::string output = response.dump() + "\n";
    ssize_t bytes_sent = send(client_sock, output.c_str(), output.size(), 0);  // 데이터를 클라이언트에 전송
    
    if (bytes_sent == -1) {  // 에러가 발생한 경우
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cerr << "[ERROR] Failed to send response, errno: " << errno << ", strerror: " << strerror(errno) << std::endl;
        
        if (errno == EPIPE) {  // Broken pipe 오류가 발생했을 경우
            std::cerr << "[ERROR] Broken pipe detected. Client may have disconnected." << std::endl;
        } else if (errno == ECONNRESET) {  // Connection reset by peer
            std::cerr << "[ERROR] Connection reset by peer." << std::endl;
        } else {
            std::cerr << "[ERROR] Unknown error occurred while sending data." << std::endl;
        }
    } else {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "[INFO] Sent response to client. Bytes sent: " << bytes_sent << std::endl;
    }
}

// MariaDB에 연결하는 함수
MYSQL* connect_db() {
    MYSQL* conn = mysql_init(nullptr);
    if (!mysql_real_connect(conn, "10.10.20.109", "LBH", "1234", "FIRE_DB", 0, NULL, 0)) {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cerr << "[ERROR] MariaDB connection failed: " << mysql_error(conn) << std::endl;
        return nullptr;  // 연결 실패 시 nullptr 반환
    }
    return conn;
}


std::string base64_decode(const std::string& encoded) {
    std::string decoded;
    BIO *bio, *b64;

    int decode_len = (encoded.length() * 3) / 4;  // Estimate the decoded length

    decoded.resize(decode_len);
    bio = BIO_new_mem_buf((void*)encoded.c_str(), -1);
    b64 = BIO_new(BIO_f_base64());
    bio = BIO_push(b64, bio);

    int decoded_len = BIO_read(bio, &decoded[0], encoded.size());
    decoded.resize(decoded_len);

    BIO_free_all(bio);
    return decoded;
}

std::string base64_encode(const std::vector<unsigned char>& data) {
    // OpenSSL을 사용하여 Base64 인코딩
    BIO *bio, *b64;
    BUF_MEM *buffer_ptr;
    
    b64 = BIO_new(BIO_f_base64());      // Base64 필터 생성
    bio = BIO_new(BIO_s_mem());         // 메모리 기반 BIO 생성
    bio = BIO_push(b64, bio);           // Base64 필터를 메모리 BIO에 추가

    // 데이터를 Base64로 인코딩
    BIO_write(bio, data.data(), data.size());
    BIO_flush(bio);

    // 인코딩된 데이터를 버퍼에 저장
    BIO_get_mem_ptr(bio, &buffer_ptr);

    // Base64 인코딩된 데이터를 반환
    std::string result(buffer_ptr->data, buffer_ptr->length);
    
    BIO_free_all(bio);
    
    return result;
}

// 이미지 파일을 Base64로 인코딩
// 파일 경로를 받아 파일 내용을 읽고 Base64로 인코딩하는 함수
std::string encode_file_to_base64(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cerr << "[ERROR] Cannot open file for reading: " << file_path << std::endl;
        return "";
    }

    std::ostringstream os;
    os << file.rdbuf();
    std::string file_content = os.str();

    BIO *bio, *b64;
    BUF_MEM *buffer_ptr;

    b64 = BIO_new(BIO_f_base64());
    bio = BIO_new(BIO_s_mem());
    bio = BIO_push(b64, bio);

    BIO_set_flags(bio, BIO_FLAGS_BASE64_NO_NL); // 결과물에 개행 문자를 추가하지 않음

    BIO_write(bio, file_content.data(), file_content.length());
    BIO_flush(bio);

    BIO_get_mem_ptr(bio, &buffer_ptr);
    std::string result(buffer_ptr->data, buffer_ptr->length);
    BIO_free_all(bio);

    return result;
}

// handle_json_message 함수
void handle_json_message(const std::string& message, int client_sock) {
    try {
        auto j = json::parse(message);
        if (!j.contains("signal")) {
            send_json(client_sock, {{"status", "error"}, {"message", "'signal' field missing"}});
            return;
        }
        std::string signal = j["signal"];
        MYSQL* conn = connect_db();
        if (!conn) return;  // DB 연결 실패 시 리턴
    
        if (signal == "fire") {
            cout << "here!!" << endl;
            string img_base64 = j["picture"];
            string cctv_num = j["camera"];
            string addr;
            string f_id;
            string time = getCurrentTimestamp();
            string img_time = getCurrentDateOnly();
            int police_socket = 0;
            int fire_station_socket = 0;

            cout << "cctv_name: " << cctv_num << endl;

            // CCTV 정보 조회
            string cctv_query = "SELECT CCTV_PLACE FROM CCTV_INFO WHERE CCTV_NAME = '" + cctv_num + "'";

            if (mysql_query(conn, cctv_query.c_str()) == 0) {
                MYSQL_RES* cctvnum_result = mysql_store_result(conn);
                MYSQL_ROW row = mysql_fetch_row(cctvnum_result);

                addr = row[0];

                mysql_free_result(cctvnum_result);
            }
            cout << "주소: " << addr << endl;

            // Base64 디코딩 (std::vector<unsigned char> 반환)
            std::string decoded_str = base64_decode(img_base64, false);  // 디코딩된 문자열
            std::vector<unsigned char> img_data(decoded_str.begin(), decoded_str.end());  // std::vector<unsigned char>로 변환

            // 저장할 디렉토리 경로 설정
            std::string save_dir = "./images/";  // 상대 경로로 설정

            std::string filename = save_dir+cctv_num+"fire_image_" + img_time + ".jpg";

            // 이미지를 파일로 저장
            std::ofstream out_file(filename, std::ios::binary);  // 이미지 파일 이름을 동적으로 설정
            out_file.write(reinterpret_cast<char*>(img_data.data()), img_data.size());  // 디코딩된 데이터 저장
            out_file.close();

            // 쿼리 준비
            string insert_fire_info = "INSERT INTO FIRE_INFO(CCTV_NAME,IMG, ADDRESS, TIME) VALUES ('"+cctv_num+"','"+filename+"','"+addr+"','"+time+"')";
            
            if (mysql_query(conn, insert_fire_info.c_str())==0)
            {
                cout << "DB에 화재 정보 저장 완료" << endl;
            }
            else
            {
                cout << "DB 저장 실패!" <<endl;
            }
            
            if (cctv_num == "cam01")
            {
                police_socket = get_socket_for_ip("10.10.20.116");
                fire_station_socket = get_socket_for_ip("10.10.20.107");
            }
            else if (cctv_num == "cam03")
            {
                police_socket = get_socket_for_ip("10.10.20.116");
                fire_station_socket = get_socket_for_ip("10.10.20.107");
            }

            string img_info = encode_file_to_base64(filename);
            // cout<< base64_encode(filename) << endl;

            if (police_socket > 0 && fire_station_socket > 0)
            {
                send_json(police_socket, {{"signal", "response_request"}, {"img", img_info}, {"place", addr}, {"time", time}});
                send_json(fire_station_socket, {{"signal", "response_request"}, {"img", img_info}, {"place", addr}, {"time", time}});
                cout << "경찰서&소방서 전송 완료!" << endl;
            }
            else if(police_socket > 0 || fire_station_socket <= 0)
            {
                send_json(police_socket, {{"signal", "response_request"}, {"img", img_info}, {"place", addr}, {"time", time}});
                cout << "경찰서 전송 완료!" << endl;
            }
            else if(police_socket <= 0 || fire_station_socket > 0)
            {
                send_json(fire_station_socket, {{"signal", "response_request"}, {"img", img_info}, {"place", addr}, {"time", time}});
                cout << "소방서 전송 완료!" << endl;
            }
            else if(police_socket <= 0 && fire_station_socket <= 0)
            {
                cout << "경찰서와 소방서가 연결 되어 있지 않습니다." << endl;
                return;
            }
        }
        else if (signal == "cam01" || signal == "cam03")
        {
            cout << "[INFO] '" << signal << "' signal received." << endl;
            string cam_name = j.value("signal", "");
            json response_data = json::array();  // 응답 데이터를 담을 배열을 초기화

            // 쿼리 수정: CCTV_NAME에 해당하는 모든 데이터를 가져오기
            string query = "SELECT CCTV_NAME, IMG, ADDRESS, TIME FROM FIRE_INFO WHERE CCTV_NAME = ?";
            MYSQL_STMT* stmt = mysql_stmt_init(conn);   
            mysql_stmt_prepare(stmt, query.c_str(), query.length());

            MYSQL_BIND bind[1];
            memset(bind, 0, sizeof(bind));
            bind[0].buffer_type = MYSQL_TYPE_STRING;
            bind[0].buffer = (char*)cam_name.c_str();
            bind[0].buffer_length = cam_name.length();
            mysql_stmt_bind_param(stmt, bind);

            mysql_stmt_execute(stmt);

            char db_cam_name[255], db_img_path[255], db_address[255], db_time[255];
            MYSQL_BIND result_bind[4];
            memset(result_bind, 0, sizeof(result_bind));
            result_bind[0].buffer_type = MYSQL_TYPE_STRING;
            result_bind[0].buffer = db_cam_name;
            result_bind[0].buffer_length = sizeof(db_cam_name);
            result_bind[1].buffer_type = MYSQL_TYPE_STRING;
            result_bind[1].buffer = db_img_path;
            result_bind[1].buffer_length = sizeof(db_img_path);
            result_bind[2].buffer_type = MYSQL_TYPE_STRING;
            result_bind[2].buffer = db_address;
            result_bind[2].buffer_length = sizeof(db_address);
            result_bind[3].buffer_type = MYSQL_TYPE_STRING;
            result_bind[3].buffer = db_time;
            result_bind[3].buffer_length = sizeof(db_time);

            mysql_stmt_bind_result(stmt, result_bind);

            // 결과가 있을 경우 계속해서 데이터를 가져와서 response_data에 추가
            while (mysql_stmt_fetch(stmt) == 0) {
                std::string encoded_image = encode_file_to_base64(db_img_path);  // 이미지 경로를 Base64로 인코딩
                json cam_info = {
                    {"camera", db_cam_name},
                    {"picture", encoded_image},
                    {"address", db_address},
                    {"time", db_time}
                };
                response_data.push_back(cam_info);  // response_data 배열에 각 카메라 정보 추가
            }

            if (!response_data.empty()) {
                // 데이터가 있을 경우
                cout<< "전송 완료! : " << response_data << endl;
                send_json(client_sock, {{"signal", "cam_result"}, {"cam_info", response_data}});
            } else {
                // 데이터가 없는 경우
                cout<< "자료 없음!" << endl;
                send_json(client_sock, {{"signal", "cam_result"}, {"cam_info", {{"message", "자료가 없습니다."}}}});
            }

            mysql_stmt_close(stmt);  // 쿼리 실행 후 반드시 stmt를 닫음
        }
        else if (signal == "police_response_completed")
        {
            cout << "경찰 출동 완료!" << endl;
        }
        else if (signal == "fire_station_response_completed")
        {
            cout << "소방서 출동 완료!" << endl;
        }
        else 
        {
            send_json(client_sock, {{"status", "error"}, {"message", "Unknown signal"}});
        }


        mysql_close(conn);
    } catch (const json::parse_error& e) {
        send_json(client_sock, {{"status", "error"}, {"message", string("JSON parse error: ") + e.what()}});
    } catch (const json::type_error& e) {
        send_json(client_sock, {{"status", "error"}, {"message", string("JSON type error: ") + e.what()}});
    }
}

// 클라이언트와의 연결을 처리하는 함수
void client_worker(int client_sock, std::string client_ip) {
    char buffer[BUFFER_SIZE];
    std::string recv_buffer;
    size_t expected_image_size = 0;
    std::ofstream image_file;
    bool receiving_image = false;

    {
        std::lock_guard<std::mutex> lock(ip_map_mutex);
        ip_to_socket[client_ip] = client_sock;
        std::lock_guard<std::mutex> lock2(cout_mutex);
        std::cout << "[INFO] Connected IP: " << client_ip << std::endl;
    }

    while (true) {
        ssize_t len = recv(client_sock, buffer, sizeof(buffer), 0);
        if (len <= 0) {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cerr << "[ERROR] Connection closed or recv() failed!" << std::endl;
            break;
        }

        // 이미지 수신 중일 때
        if (receiving_image) {
            image_file.write(buffer, len);
            expected_image_size -= len;
            if (expected_image_size <= 0) {
                image_file.close();
                receiving_image = false;
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "[INFO] 이미지 수신 완료!" << std::endl;
            }
            continue;
        }

        recv_buffer.append(buffer, len);
        size_t pos;
        while ((pos = recv_buffer.find('\n')) != std::string::npos) {
            std::string json_str = recv_buffer.substr(0, pos);
            recv_buffer.erase(0, pos + 1);

            try {
                json j = json::parse(json_str);
                std::string signal = j["signal"];

                // 이미지 업로드 신호 처리
                if (signal == "upload_image") {
                    std::string filename = j["file_name"];
                    expected_image_size = j["file_size"];
                    image_file.open("./received_" + filename, std::ios::binary);
                    receiving_image = true;

                    std::lock_guard<std::mutex> lock(cout_mutex);
                    std::cout << "[INFO] 이미지 수신 시작 (" << filename << ", " << expected_image_size << " bytes)" << std::endl;
                    continue;
                }

                // 일반적인 JSON 메시지 처리
                handle_json_message(json_str, client_sock);
            } catch (...) {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cerr << "[ERROR] JSON 파싱 오류!" << std::endl;
            }
        }
    }

    close(client_sock);
    {
        std::lock_guard<std::mutex> lock(ip_map_mutex);
        ip_to_socket.erase(client_ip);
    }
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << "[INFO] Client disconnected: " << client_ip << std::endl;
}

// 서버를 시작하고 연결을 처리하는 함수
void start_server() {
    int server_fd;
    struct sockaddr_in server_addr;
    int opt = 1;

    // 소켓 생성
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);  // 소켓 생성 실패 시 서버 종료
    }

    // 주소 재사용 설정
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt failed");
        close(server_fd);
        exit(EXIT_FAILURE);  // setsockopt 실패 시 종료
    }

    // 서버 주소 설정
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);

    // 서버 주소 바인딩
    if (bind(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        close(server_fd);
        exit(EXIT_FAILURE);  // bind 실패 시 서버 종료
    }

    // 연결 수신 대기 시작
    if (listen(server_fd, 10) < 0) {
        perror("listen failed");
        close(server_fd);
        exit(EXIT_FAILURE);  // listen 실패 시 서버 종료
    }

    std::cout << "[INFO] Server listening on port " << PORT << std::endl;

    // 클라이언트 연결 수락 루프
    while (true) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_sock = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);

        if (client_sock < 0) {
            perror("accept failed");
            continue;  // accept 실패 시 계속해서 새로운 클라이언트 연결 대기
        }

        char ip_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(client_addr.sin_addr), ip_str, INET_ADDRSTRLEN);
        std::string client_ip(ip_str);

        // 클라이언트 요청을 처리하는 스레드 생성
        std::thread(client_worker, client_sock, client_ip).detach();
    }

    close(server_fd);  // 서버 종료 시 소켓 닫기
}

// 주어진 IP에 대한 소켓을 찾는 함수
int get_socket_for_ip(const std::string& ip) {
    std::lock_guard<std::mutex> lock(ip_map_mutex);
    auto it = ip_to_socket.find(ip);
    if (it != ip_to_socket.end()) {
        std::cout << "[INFO] Socket for IP " << ip << " : " << it->second << std::endl;
        return it->second;
    } else {
        std::cout << "[WARN] No socket found for IP " << ip << std::endl;
        return -1;  // IP에 해당하는 소켓이 없으면 -1 반환
    }
}

// 현재 시간을 문자열로 반환하는 함수
string getCurrentTimeString() {
    std::time_t now = std::time(nullptr);
    char buf[100];
    std::strftime(buf, sizeof(buf), "%F %T", std::localtime(&now));
    return std::string(buf);
}

// 현재 타임스탬프를 반환하는 함수
string getCurrentTimestamp() {
    std::time_t now = std::time(nullptr);
    std::tm* local_time = std::localtime(&now);

    std::ostringstream oss;
    oss << std::put_time(local_time, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

// 년-월-일 표시
// string getCurrentDateOnly() {
//     std::time_t now = std::time(nullptr);
//     std::tm* local_time = std::localtime(&now);

//     std::ostringstream oss;
//     oss << std::put_time(local_time, "%Y-%m-%d%H:%M:%S");  // 시간 제외
//     return oss.str();
// }
// 현재 날짜만 반환하는 함수
std::string getCurrentDateOnly() {
    using namespace std::chrono;
    
    // 현재 시간 얻기
    auto now = system_clock::now();
    
    // 현재 시간의 초와 나노초를 얻기 위한 계산
    auto time_since_epoch = now.time_since_epoch();
    std::chrono::seconds seconds = duration_cast<std::chrono::seconds>(time_since_epoch); // 초 단위로 변환
    auto nanoseconds = duration_cast<std::chrono::nanoseconds>(time_since_epoch) - seconds; // 나노초 단위로 남은 값 추출

    // 시간 출력
    std::time_t now_time = system_clock::to_time_t(now);
    std::tm* local_time = std::localtime(&now_time);

    // 결과 문자열을 스트림으로 구성
    std::ostringstream oss;
    oss << std::put_time(local_time, "%Y-%m-%d%H:%M:%S");

    // 소수점 초(나노초 단위)를 추가
    oss << '.' << std::setw(9) << std::setfill('0') << nanoseconds.count();

    return oss.str();
}

// 주어진 문자열에서 첫 번째 UTF-8 문자를 반환하는 함수
string get_first_utf8_char(const std::string& str) {
    if (str.empty()) return "";
    unsigned char lead = static_cast<unsigned char>(str[0]);
    size_t len = 1;
    if ((lead & 0xF8) == 0xF0) len = 4;
    else if ((lead & 0xF0) == 0xE0) len = 3;
    else if ((lead & 0xE0) == 0xC0) len = 2;
    return str.substr(0, len);
}


// 서버 시작 함수
int main() {
    start_server();
    return 0;
}
