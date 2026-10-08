/*
 * ============================================================
 * YOLO26 MERGED PIPELINE
 * ============================================================
 *
 * Combines:
 *   - File 1: snapshot capture + JSON detection events
 *   - File 2: live annotated RTSP re-stream
 *
 * RTSP INPUT:
 *   Read automatically from:
 *   /var/lib/camera-discovery/cameras.json
 *
 * SNAPSHOTS:
 *   Only vehicles (car, motorcycle, bus, truck) that cross the red
 *   crosswalk line are saved. Each tracked vehicle is saved once
 *   (one vehicle = one JPG) into SNAPSHOT_DIR/vehicles/.
 *
 * Pipeline:
 *
 *   nvurisrcbin -> streammux -> nvinfer -> nvtracker -> nvdsosd -> tee
 *        |                                             |-- queue -> fakesink
 *        |                                             |-- queue -> nvvideoconvert ->
 *        |                                             |            capsfilter -> nvjpegenc ->
 *        |                                             |            appsink
 *        |                                             |-- queue -> nvrtspoutsinkbin
 *
 * ============================================================
 */

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <glib.h>
#include <glib-unix.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <csignal>
#include <string>
#include <map>
#include <mutex>
#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <cctype>

#include <nlohmann/json.hpp>

#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

using json = nlohmann::json;


/*
 * ============================================================
 * Configuration
 * ============================================================
 */

#define SNAPSHOT_DIR              "/root/video_recorder/snapshots"
#define EVENT_DIR                 "/root/video_recorder/events/pending"
#define DEEPSTREAM_CONFIG         "/opt/radian/ai/config/config_infer_primary_yolo26.txt"
#define VEHICLE_SNAPSHOT_DIR      SNAPSHOT_DIR "/vehicles"

/* Must match nvstreammux width/height in main() */
#define MUX_WIDTH                 1920
#define MUX_HEIGHT                1080
#define CROSSING_MARGIN           15.0f

/* nvtracker (needed to give each vehicle a stable ID) */
#define TRACKER_LIB   "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
#define TRACKER_CONFIG "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"
#define TRACKER_WIDTH  640   /* must be multiple of 32 */
#define TRACKER_HEIGHT 384   /* must be multiple of 32 */
#define RTSP_OUTPUT_URL_HINT      "rtsp://192.168.1.155:8554/ai"

/* Camera discovery output */
#define CAMERA_JSON_FILE          "/var/lib/camera-discovery/cameras.json"
#define CROSSWALK_JSON_FILE "/var/lib/radian/crosswalk/crosswalk.json"


/*
 * ============================================================
 * Shared state — touched from multiple threads, hence mutex.
 * ============================================================
 */

struct SnapshotData
{
    std::vector<unsigned char> jpeg_data;
    bool valid = false;
};

static SnapshotData latest_snapshot;
static std::mutex snapshot_mutex;

/*
 * Line-crossing state. Only touched from the single streaming thread
 * that runs infer_src_probe, so no mutex is needed.
 */
struct TrackState
{
    float last_y = 0.0f;   /* bottom edge of the bbox in previous frame */
    int last_frame = 0;
    bool initialized = false;
    bool was_above = false;
    bool was_below = false;
};

static std::unordered_map<guint64, TrackState> track_states;
static std::unordered_set<guint64> captured_ids;   /* already saved */

static GMainLoop *main_loop = nullptr;


/*
 * ============================================================
 * Read RTSP URL from camera-discovery JSON
 * ============================================================
 *
 * Expected JSON:
 *
 * {
 *     "version": 1,
 *     "updated_at": "...",
 *     "cameras": [
 *         {
 *             "id": "camera_001",
 *             "mac": "...",
 *             "ip": "10.10.10.156",
 *             "status": "online",
 *             "rtsp": {
 *                 "valid": true,
 *                 "url": "rtsp://..."
 *             }
 *         }
 *     ]
 * }
 *
 * The first camera that is:
 *
 *     status = online
 *     rtsp.valid = true
 *
 * will be selected.
 *
 * ============================================================
 */

struct CrosswalkLineConfig
{
    bool detected = false;
    int line_y = -1;
    int frame_width = 0;
    int frame_height = 0;
};

static CrosswalkLineConfig crosswalk_config;

static bool load_crosswalk_line_config()
{
    std::ifstream file(CROSSWALK_JSON_FILE);
    if (!file.is_open())
    {
        std::cerr << "[CROSSWALK] Cannot open " << CROSSWALK_JSON_FILE << std::endl;
        return false;
    }

    try
    {
        nlohmann::json j;
        file >> j;

        if (!j.contains("crosswalk_detected") ||
            !j.contains("line_y") ||
            !j.contains("frame_width") ||
            !j.contains("frame_height"))
        {
            std::cerr << "[CROSSWALK] Invalid JSON: required fields missing" << std::endl;
            return false;
        }

        crosswalk_config.detected = j["crosswalk_detected"].get<bool>();
        crosswalk_config.line_y = j["line_y"].get<int>();
        crosswalk_config.frame_width = j["frame_width"].get<int>();
        crosswalk_config.frame_height = j["frame_height"].get<int>();

        std::cout << "[CROSSWALK] JSON loaded" << std::endl;
        std::cout << "[CROSSWALK] detected = " << crosswalk_config.detected << std::endl;
        std::cout << "[CROSSWALK] line_y = " << crosswalk_config.line_y << std::endl;
        std::cout << "[CROSSWALK] frame = "
                  << crosswalk_config.frame_width << "x"
                  << crosswalk_config.frame_height << std::endl;

        if (!crosswalk_config.detected ||
            crosswalk_config.line_y < 0 ||
            crosswalk_config.frame_width <= 0 ||
            crosswalk_config.frame_height <= 0)
        {
            std::cerr << "[CROSSWALK] No valid crosswalk line configuration" << std::endl;
            return false;
        }

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[CROSSWALK] JSON parse error: " << e.what() << std::endl;
        return false;
    }
}


/*
 * Y position of the red line in the pipeline (streammux) coordinate
 * space. Object boxes are in this same space, so drawing and crossing
 * detection always agree.
 */
static int compute_line_y(const NvDsFrameMeta *frame_meta)
{
    int h = (frame_meta->pipeline_height > 0)
        ? static_cast<int>(frame_meta->pipeline_height)
        : MUX_HEIGHT;

    int y = crosswalk_config.line_y;

    if (crosswalk_config.frame_height > 0 &&
        crosswalk_config.frame_height != h)
    {
        y = static_cast<int>(
            static_cast<double>(crosswalk_config.line_y) *
            static_cast<double>(h) /
            static_cast<double>(crosswalk_config.frame_height));
    }

    if (y < 0)
        y = 0;

    if (y >= h)
        y = h - 1;

    return y;
}


static std::string get_rtsp_url_from_json()
{
    std::ifstream file(CAMERA_JSON_FILE);

    if (!file.is_open())
    {
        std::cerr << "ERROR: Cannot open camera JSON file: "
                  << CAMERA_JSON_FILE << std::endl;

        return "";
    }

    try
    {
        json data;
        file >> data;

        if (!data.contains("cameras") ||
            !data["cameras"].is_array())
        {
            std::cerr << "ERROR: Invalid camera JSON: "
                      << "'cameras' array not found."
                      << std::endl;

            return "";
        }

        for (const auto &camera : data["cameras"])
        {
            /*
             * Check camera status
             */
            if (!camera.contains("status") ||
                camera["status"] != "online")
            {
                continue;
            }

            /*
             * Check RTSP section
             */
            if (!camera.contains("rtsp") ||
                !camera["rtsp"].is_object())
            {
                continue;
            }

            /*
             * Check RTSP validity
             */
            if (!camera["rtsp"].contains("valid") ||
                !camera["rtsp"]["valid"].get<bool>())
            {
                continue;
            }

            /*
             * Check RTSP URL
             */
            if (!camera["rtsp"].contains("url") ||
                !camera["rtsp"]["url"].is_string())
            {
                continue;
            }

            std::string camera_id =
                camera.value("id", "unknown");

            std::string rtsp_url =
                camera["rtsp"]["url"].get<std::string>();

            if (rtsp_url.empty())
            {
                continue;
            }

            std::cout << "========================================"
                      << std::endl;

            std::cout << "Camera ID : "
                      << camera_id
                      << std::endl;

            std::cout << "Camera IP : "
                      << camera.value("ip", "unknown")
                      << std::endl;

            std::cout << "RTSP URL  : "
                      << rtsp_url
                      << std::endl;

            std::cout << "========================================"
                      << std::endl;

            return rtsp_url;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR: Failed to parse "
                  << CAMERA_JSON_FILE
                  << ": "
                  << e.what()
                  << std::endl;

        return "";
    }

    std::cerr << "ERROR: No valid online camera found in "
              << CAMERA_JSON_FILE
              << std::endl;

    return "";
}


/*
 * ============================================================
 * Small helpers
 * ============================================================
 */

static std::string get_timestamp()
{
    std::time_t now = std::time(nullptr);
    std::tm tm_now;
    gmtime_r(&now, &tm_now);

    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%Y-%m-%dT%H:%M:%S");

    return oss.str();
}


static bool save_jpeg_data(
    const std::vector<unsigned char> &data,
    const std::string &filename)
{
    if (data.empty())
    {
        g_printerr("JPEG data is empty\n");
        return false;
    }

    std::ofstream file(filename, std::ios::binary);

    if (!file)
    {
        g_printerr(
            "Failed to open snapshot file: %s\n",
            filename.c_str());

        return false;
    }

    file.write(
        reinterpret_cast<const char *>(data.data()),
        static_cast<std::streamsize>(data.size()));

    file.close();

    if (!file)
    {
        g_printerr(
            "Failed while writing snapshot: %s\n",
            filename.c_str());

        return false;
    }

    return true;
}


static std::string safe_filename_component(
    const std::string &value)
{
    std::string result;

    for (char c : value)
    {
        unsigned char uc =
            static_cast<unsigned char>(c);

        result +=
            (std::isalnum(uc) ||
             c == '_' ||
             c == '-') ? c : '_';
    }

    if (result.empty())
        result = "object";

    return result;
}


static std::string json_escape(
    const std::string &value)
{
    std::string result;

    for (char c : value)
    {
        switch (c)
        {
            case '\\':
                result += "\\\\";
                break;

            case '"':
                result += "\\\"";
                break;

            case '\n':
                result += "\\n";
                break;

            case '\r':
                result += "\\r";
                break;

            case '\t':
                result += "\\t";
                break;

            default:
                result += c;
                break;
        }
    }

    return result;
}


static bool save_detection_event(
    const std::string &object_name,
    float confidence,
    std::time_t detection_time,
    const std::string &timestamp,
    const std::string &snapshot_path)
{
    std::ostringstream filename;

    filename << EVENT_DIR << "/"
             << safe_filename_component(object_name)
             << "_"
             << timestamp
             << ".json";

    std::ofstream file(filename.str());

    if (!file)
    {
        g_printerr(
            "Failed to create detection event: %s\n",
            filename.str().c_str());

        return false;
    }

    file << "{\n"
         << "  \"object\": \""
         << json_escape(object_name)
         << "\",\n"

         << "  \"confidence\": "
         << std::fixed
         << std::setprecision(4)
         << confidence
         << ",\n"

         << "  \"timestamp\": \""
         << timestamp
         << "\",\n"

         << "  \"unix_timestamp\": "
         << static_cast<long long>(detection_time)
         << ",\n"

         << "  \"snapshot\": \""
         << json_escape(snapshot_path)
         << "\"\n"

         << "}\n";

    file.close();

    if (!file)
    {
        g_printerr(
            "Failed while writing detection event: %s\n",
            filename.str().c_str());

        return false;
    }

    g_print(
        "EVENT SAVED: %s\n",
        filename.str().c_str());

    return true;
}


/*
 * Save ONE jpg for ONE vehicle.
 * This function is called ONLY after a confirmed vehicle line crossing.
 */
static bool save_vehicle_snapshot(
    const std::string &object_name,
    float confidence,
    guint64 track_id)
{
    std::time_t now = std::time(nullptr);
    std::vector<unsigned char> jpeg_data;

    {
        std::lock_guard<std::mutex> lock(snapshot_mutex);

        if (!latest_snapshot.valid ||
            latest_snapshot.jpeg_data.empty())
        {
            g_printerr("No JPEG frame available for snapshot\n");
            return false;
        }

        jpeg_data = latest_snapshot.jpeg_data;
    }

    std::string timestamp = get_timestamp();

    if (g_mkdir_with_parents(VEHICLE_SNAPSHOT_DIR, 0755) != 0)
    {
        g_printerr(
            "Failed to create vehicle snapshot directory: %s\n",
            VEHICLE_SNAPSHOT_DIR);
        return false;
    }

    std::ostringstream filename;

    filename << VEHICLE_SNAPSHOT_DIR << "/"
             << safe_filename_component(object_name)
             << "_id" << track_id
             << "_" << std::fixed << std::setprecision(2) << confidence
             << "_" << timestamp
             << ".jpg";

    if (!save_jpeg_data(jpeg_data, filename.str()))
    {
        return false;
    }

    g_print("SNAPSHOT SAVED: %s\n", filename.str().c_str());

    save_detection_event(
        object_name,
        confidence,
        now,
        timestamp,
        filename.str());

    return true;
}


/*
 * ============================================================
 * appsink callback
 * ============================================================
 */

static GstFlowReturn on_new_sample(
    GstAppSink *appsink,
    gpointer)
{
    GstSample *sample =
        gst_app_sink_pull_sample(appsink);

    if (!sample)
        return GST_FLOW_ERROR;

    GstBuffer *buffer =
        gst_sample_get_buffer(sample);

    if (!buffer)
    {
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }

    GstMapInfo map;

    if (gst_buffer_map(
            buffer,
            &map,
            GST_MAP_READ))
    {
        std::vector<unsigned char> jpeg(
            map.data,
            map.data + map.size);

        {
            std::lock_guard<std::mutex>
                lock(snapshot_mutex);

            latest_snapshot.jpeg_data =
                jpeg;

            latest_snapshot.valid =
                true;
        }

        gst_buffer_unmap(
            buffer,
            &map);
    }

    gst_sample_unref(sample);

    return GST_FLOW_OK;
}


/*
 * ============================================================
 */

static bool is_vehicle_class(const std::string &name)
{
    static const std::set<std::string> vehicles = {
        "car",
        "motorcycle",
        "bus",
        "truck"
    };

    return vehicles.count(name) > 0;
}

static GstPadProbeReturn infer_src_probe(
    GstPad *,
    GstPadProbeInfo *info,
    gpointer)
{
    if (!crosswalk_config.detected ||
        crosswalk_config.line_y < 0 ||
        crosswalk_config.frame_height <= 0)
    {
        return GST_PAD_PROBE_OK;
    }

    GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);

    if (!buf)
        return GST_PAD_PROBE_OK;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    if (!batch_meta)
        return GST_PAD_PROBE_OK;

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list;
         l_frame;
         l_frame = l_frame->next)
    {
        auto *frame_meta = static_cast<NvDsFrameMeta *>(l_frame->data);

        if (!frame_meta)
            continue;

        const int line_y = compute_line_y(frame_meta);
        const int frame_num = static_cast<int>(frame_meta->frame_num);

        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list;
             l_obj;
             l_obj = l_obj->next)
        {
            auto *obj_meta = static_cast<NvDsObjectMeta *>(l_obj->data);

            if (!obj_meta)
                continue;

            if (obj_meta->object_id == UNTRACKED_OBJECT_ID)
                continue;

            const std::string object_name = obj_meta->obj_label;

            if (!is_vehicle_class(object_name))
                continue;

            const guint64 id = obj_meta->object_id;

            if (captured_ids.count(id))
                continue;

            const float y =
                obj_meta->rect_params.top +
                obj_meta->rect_params.height;

            auto it = track_states.find(id);

            if (it == track_states.end())
            {
                TrackState state;
                state.last_y = y;
                state.last_frame = frame_num;
                state.initialized = true;
                state.was_above =
                    (y < static_cast<float>(line_y) - CROSSING_MARGIN);
                state.was_below =
                    (y > static_cast<float>(line_y) + CROSSING_MARGIN);

                track_states[id] = state;
                continue;
            }

            TrackState &state = it->second;

            const float above_limit =
                static_cast<float>(line_y) - CROSSING_MARGIN;

            const float below_limit =
                static_cast<float>(line_y) + CROSSING_MARGIN;

            bool crossed = false;

            if (state.was_above && y > below_limit)
            {
                crossed = true;
            }
            else if (state.was_below && y < above_limit)
            {
                crossed = true;
            }

            if (y < above_limit)
            {
                state.was_above = true;
                state.was_below = false;
            }
            else if (y > below_limit)
            {
                state.was_below = true;
                state.was_above = false;
            }

            state.last_y = y;
            state.last_frame = frame_num;

            if (!crossed)
                continue;

            std::cout
                << "VEHICLE CROSSED RED LINE: "
                << object_name
                << " id=" << id
                << " confidence=" << obj_meta->confidence
                << std::endl;

            if (save_vehicle_snapshot(
                    object_name,
                    obj_meta->confidence,
                    id))
            {
                captured_ids.insert(id);
            }
        }

        if (track_states.size() > 1000)
        {
            for (auto it = track_states.begin();
                 it != track_states.end();)
            {
                if (frame_num - it->second.last_frame > 900)
                {
                    it = track_states.erase(it);
                }
                else
                {
                    ++it;
                }
            }
        }
    }

    return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn crosswalk_line_probe(
    GstPad *pad,
    GstPadProbeInfo *info,
    gpointer user_data)
{
    if (!crosswalk_config.detected)
        return GST_PAD_PROBE_OK;

    GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf)
        return GST_PAD_PROBE_OK;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta)
        return GST_PAD_PROBE_OK;

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list;
         l_frame != nullptr;
         l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta =
            static_cast<NvDsFrameMeta *>(l_frame->data);

        if (!frame_meta)
            continue;

        int frame_width = (frame_meta->pipeline_width > 0)
            ? static_cast<int>(frame_meta->pipeline_width)
            : MUX_WIDTH;

        int line_y = compute_line_y(frame_meta);

        NvDsDisplayMeta *display_meta =
            nvds_acquire_display_meta_from_pool(batch_meta);

        if (!display_meta)
            continue;

        display_meta->num_lines = 1;

        NvOSD_LineParams &line = display_meta->line_params[0];

        line.x1 = 0;
        line.y1 = line_y;
        line.x2 = frame_width;
        line.y2 = line_y;
        line.line_width = 5;

        line.line_color.red = 1.0;
        line.line_color.green = 0.0;
        line.line_color.blue = 0.0;
        line.line_color.alpha = 1.0;

        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }

    return GST_PAD_PROBE_OK;
}


static void on_pad_added(
    GstElement *,
    GstPad *new_pad,
    gpointer user_data)
{
    GstElement *streammux =
        GST_ELEMENT(user_data);

    GstCaps *caps =
        gst_pad_get_current_caps(new_pad);

    if (!caps)
        caps =
            gst_pad_query_caps(
                new_pad,
                nullptr);

    if (!caps)
        return;

    GstStructure *structure =
        gst_caps_get_structure(caps, 0);

    const gchar *name =
        gst_structure_get_name(structure);

    std::cout
        << "nvurisrcbin pad added: "
        << name
        << std::endl;

    if (!g_str_has_prefix(
            name,
            "video/"))
    {
        gst_caps_unref(caps);
        return;
    }

    GstPad *sink_pad =
        gst_element_request_pad_simple(
            streammux,
            "sink_0");

    if (!sink_pad)
    {
        std::cerr
            << "ERROR: Cannot get "
               "nvstreammux sink_0"
            << std::endl;

        gst_caps_unref(caps);
        return;
    }

    if (gst_pad_is_linked(sink_pad))
    {
        gst_object_unref(sink_pad);
        gst_caps_unref(caps);
        return;
    }

    GstPadLinkReturn ret =
        gst_pad_link(
            new_pad,
            sink_pad);

    if (ret == GST_PAD_LINK_OK)
    {
        std::cout
            << "Video linked to nvstreammux."
            << std::endl;
    }
    else
    {
        std::cerr
            << "ERROR: Failed to link video "
               "to nvstreammux. Return="
            << ret
            << std::endl;
    }

    gst_object_unref(sink_pad);
    gst_caps_unref(caps);
}


/*
 * ============================================================
 * Bus callback
 * ============================================================
 */

static gboolean bus_callback(
    GstBus *,
    GstMessage *message,
    gpointer user_data)
{
    GMainLoop *loop =
        static_cast<GMainLoop *>(user_data);

    switch (GST_MESSAGE_TYPE(message))
    {
        case GST_MESSAGE_ERROR:
        {
            GError *error = nullptr;
            gchar *debug = nullptr;

            gst_message_parse_error(
                message,
                &error,
                &debug);

            std::cerr
                << "\nGStreamer ERROR from "
                << GST_OBJECT_NAME(message->src)
                << ": "
                << error->message
                << std::endl;

            if (debug)
            {
                std::cerr
                    << "Debug: "
                    << debug
                    << std::endl;
            }

            g_error_free(error);
            g_free(debug);

            g_main_loop_quit(loop);

            break;
        }

        case GST_MESSAGE_WARNING:
        {
            GError *error = nullptr;
            gchar *debug = nullptr;

            gst_message_parse_warning(
                message,
                &error,
                &debug);

            std::cerr
                << "\nGStreamer WARNING from "
                << GST_OBJECT_NAME(message->src)
                << ": "
                << error->message
                << std::endl;

            if (debug)
            {
                std::cerr
                    << "Debug: "
                    << debug
                    << std::endl;
            }

            g_error_free(error);
            g_free(debug);

            break;
        }

        case GST_MESSAGE_EOS:
        {
            std::cout
                << "\nEOS received."
                << std::endl;

            g_main_loop_quit(loop);

            break;
        }

        default:
            break;
    }

    return G_SOURCE_CONTINUE;
}


static gboolean stop_pipeline(
    gpointer user_data)
{
    g_main_loop_quit(
        static_cast<GMainLoop *>(user_data));

    return G_SOURCE_REMOVE;
}


/*
 * ============================================================
 * Tee helper
 * ============================================================
 */

static bool link_tee_branch(
    GstElement *tee,
    GstElement *target,
    const char *branch_name)
{
    GstPad *tee_src =
        gst_element_request_pad_simple(
            tee,
            "src_%u");

    GstPad *sink_pad =
        gst_element_get_static_pad(
            target,
            "sink");

    if (!tee_src || !sink_pad)
    {
        g_printerr(
            "Failed to get pads for tee branch: %s\n",
            branch_name);

        if (tee_src)
            gst_object_unref(tee_src);

        if (sink_pad)
            gst_object_unref(sink_pad);

        return false;
    }

    bool ok =
        (gst_pad_link(
            tee_src,
            sink_pad)
         == GST_PAD_LINK_OK);

    if (!ok)
    {
        g_printerr(
            "Failed to link tee -> %s\n",
            branch_name);
    }

    gst_object_unref(tee_src);
    gst_object_unref(sink_pad);

    return ok;
}


/*
 * ============================================================
 * MAIN
 * ============================================================
 */

int main(int argc, char *argv[])
{
    bool crosswalk_loaded = load_crosswalk_line_config();

    if (!crosswalk_loaded)
    {
        std::cerr << "[CROSSWALK] WARNING: fixed line will not be drawn" << std::endl;
    }
    else
    {
        std::cout << "[CROSSWALK] Fixed line enabled at Y="
                  << crosswalk_config.line_y << std::endl;
    }

    /*
     * ========================================================
     * NEW:
     * Get RTSP URL from cameras.json
     *
     * No RTSP URL is accepted from command line anymore.
     * ========================================================
     */

    std::string rtsp_url_string =
        get_rtsp_url_from_json();

    if (rtsp_url_string.empty())
    {
        std::cerr
            << "ERROR: No valid RTSP stream available."
            << std::endl;

        return 1;
    }

    const gchar *rtsp_url =
        rtsp_url_string.c_str();


    gst_init(&argc, &argv);


    if (g_mkdir_with_parents(
            SNAPSHOT_DIR,
            0755) == 0)
    {
        g_print(
            "Snapshot directory ready: %s\n",
            SNAPSHOT_DIR);
    }
    else
    {
        g_printerr(
            "Warning: could not create snapshot directory: %s\n",
            SNAPSHOT_DIR);
    }


    if (g_mkdir_with_parents(
            EVENT_DIR,
            0755) == 0)
    {
        g_print(
            "Detection event directory ready: %s\n",
            EVENT_DIR);
    }
    else
    {
        g_printerr(
            "Warning: could not create event directory: %s\n",
            EVENT_DIR);
    }


    main_loop =
        g_main_loop_new(
            nullptr,
            FALSE);


    GstElement *pipeline =
        gst_pipeline_new(
            "yolo26-merged-pipeline");


    /* Core chain */

    GstElement *source =
        gst_element_factory_make(
            "nvurisrcbin",
            "camera-source");

    GstElement *streammux =
        gst_element_factory_make(
            "nvstreammux",
            "stream-muxer");

    GstElement *infer =
        gst_element_factory_make(
            "nvinfer",
            "yolo26-inference");

    GstElement *tracker =
        gst_element_factory_make(
            "nvtracker",
            "vehicle-tracker");
    GstElement *osd =
        gst_element_factory_make(
            "nvdsosd",
            "shared-osd");

    GstElement *tee =
        gst_element_factory_make(
            "tee",
            "infer-tee");


    /* Branch A: detection-only sink */

    GstElement *detect_queue =
        gst_element_factory_make(
            "queue",
            "detect-queue");

    GstElement *fake_sink =
        gst_element_factory_make(
            "fakesink",
            "detect-sink");


    /* Branch B: snapshot */

    GstElement *snapshot_queue =
        gst_element_factory_make(
            "queue",
            "snapshot-queue");

    GstElement *converter =
        gst_element_factory_make(
            "nvvideoconvert",
            "snapshot-converter");

    GstElement *snapshot_capsfilter =
        gst_element_factory_make(
            "capsfilter",
            "snapshot-caps");

    GstElement *jpegenc =
        gst_element_factory_make(
            "nvjpegenc",
            "snapshot-jpegenc");

    GstElement *appsink_element =
        gst_element_factory_make(
            "appsink",
            "snapshot-appsink");


    /* Branch C: live RTSP re-stream */

    GstElement *rtsp_queue =
        gst_element_factory_make(
            "queue",
            "rtsp-queue");

    GstElement *rtsp_sink =
        gst_element_factory_make(
            "nvrtspoutsinkbin",
            "rtsp-output");


    if (!pipeline ||
        !source ||
        !streammux ||
        !infer ||
        !tracker ||
        !osd ||
        !tee ||
        !detect_queue ||
        !fake_sink ||
        !snapshot_queue ||
        !converter ||
        !snapshot_capsfilter ||
        !jpegenc ||
        !appsink_element ||
        !rtsp_queue ||
        !rtsp_sink)
    {
        std::cerr
            << "ERROR: Failed to create one or more "
               "GStreamer/DeepStream elements."
            << std::endl;

        if (pipeline)
            gst_object_unref(pipeline);

        g_main_loop_unref(main_loop);

        return 1;
    }


    /* --- Properties --- */

    g_object_set(
        G_OBJECT(source),
        "uri",
        rtsp_url,
        "latency",
        200,
        nullptr);


    g_object_set(
        G_OBJECT(streammux),
        "batch-size",
        1,
        "width",
        1920,
        "height",
        1080,
        "live-source",
        TRUE,
        "batched-push-timeout",
        40000,
        nullptr);


    g_object_set(
        G_OBJECT(infer),
        "config-file-path",
        DEEPSTREAM_CONFIG,
        nullptr);


    g_object_set(
        G_OBJECT(tracker),
        "ll-lib-file",
        TRACKER_LIB,
        "ll-config-file",
        TRACKER_CONFIG,
        "tracker-width",
        TRACKER_WIDTH,
        "tracker-height",
        TRACKER_HEIGHT,
        nullptr);

    g_object_set(
        G_OBJECT(osd),
        "process-mode",
        1,
        "display-bbox",
        TRUE,
        "display-text",
        TRUE,
        nullptr);


    g_object_set(
        G_OBJECT(appsink_element),
        "emit-signals",
        TRUE,
        "sync",
        FALSE,
        "max-buffers",
        1,
        "drop",
        TRUE,
        nullptr);


    GstCaps *snapshot_caps =
        gst_caps_from_string(
            "video/x-raw(memory:NVMM),"
            "format=NV12,"
            "width=1280,"
            "height=720");

    if (!snapshot_caps)
    {
        std::cerr
            << "ERROR: Failed to create snapshot caps."
            << std::endl;

        gst_object_unref(pipeline);
        g_main_loop_unref(main_loop);

        return 1;
    }


    g_object_set(
        G_OBJECT(snapshot_capsfilter),
        "caps",
        snapshot_caps,
        nullptr);

    gst_caps_unref(snapshot_caps);


    g_object_set(
        G_OBJECT(rtsp_sink),
        "idrinterval",
        30,
        "iframeinterval",
        30,
        nullptr);


    /* --- Add everything to pipeline --- */

    gst_bin_add_many(
        GST_BIN(pipeline),

        source,
        streammux,
        infer,
        tracker,
        osd,
        tee,

        detect_queue,
        fake_sink,

        snapshot_queue,
        converter,
        snapshot_capsfilter,
        jpegenc,
        appsink_element,

        rtsp_queue,
        rtsp_sink,

        nullptr);


    /* --- Static links --- */

    if (!gst_element_link_many(
            streammux,
            infer,
            tracker,
            osd,
            tee,
            nullptr))
    {
        std::cerr
            << "ERROR: Failed to link "
               "streammux -> nvinfer -> nvtracker -> "
               "nvdsosd -> tee."
            << std::endl;

        gst_object_unref(pipeline);
        g_main_loop_unref(main_loop);

        return 1;
    }


    if (!gst_element_link_many(
            detect_queue,
            fake_sink,
            nullptr))
    {
        std::cerr
            << "ERROR: Failed to link "
               "detection branch."
            << std::endl;

        gst_object_unref(pipeline);
        g_main_loop_unref(main_loop);

        return 1;
    }


    if (!gst_element_link_many(
            snapshot_queue,
            converter,
            snapshot_capsfilter,
            jpegenc,
            appsink_element,
            nullptr))
    {
        std::cerr
            << "ERROR: Failed to link "
               "snapshot branch."
            << std::endl;

        gst_object_unref(pipeline);
        g_main_loop_unref(main_loop);

        return 1;
    }


    /* rtsp_queue -> nvrtspoutsinkbin */

    {
        GstPad *rtsp_sink_pad =
            gst_element_request_pad_simple(
                rtsp_sink,
                "vsink");

        GstPad *queue_src_pad =
            gst_element_get_static_pad(
                rtsp_queue,
                "src");

        if (!rtsp_sink_pad ||
            !queue_src_pad ||
            gst_pad_link(
                queue_src_pad,
                rtsp_sink_pad)
                != GST_PAD_LINK_OK)
        {
            std::cerr
                << "ERROR: Failed to link "
                   "rtsp-queue -> "
                   "nvrtspoutsinkbin."
                << std::endl;

            if (rtsp_sink_pad)
                gst_object_unref(
                    rtsp_sink_pad);

            if (queue_src_pad)
                gst_object_unref(
                    queue_src_pad);

            gst_object_unref(pipeline);
            g_main_loop_unref(main_loop);

            return 1;
        }

        gst_object_unref(rtsp_sink_pad);
        gst_object_unref(queue_src_pad);
    }


    /* --- Tee: request 3 src pads --- */

    if (!link_tee_branch(
            tee,
            detect_queue,
            "detection queue") ||

        !link_tee_branch(
            tee,
            snapshot_queue,
            "snapshot queue") ||

        !link_tee_branch(
            tee,
            rtsp_queue,
            "rtsp queue"))
    {
        gst_object_unref(pipeline);
        g_main_loop_unref(main_loop);

        return 1;
    }


    /* --- Dynamic source pad --- */

    g_signal_connect(
        source,
        "pad-added",
        G_CALLBACK(on_pad_added),
        streammux);


    /* --- Fixed Crosswalk line probe --- */
    GstPad *osd_sink = gst_element_get_static_pad(osd, "sink");
    if (!osd_sink)
    {
        std::cerr << "[CROSSWALK] Failed to get OSD sink pad" << std::endl;
    }
    else
    {
        gst_pad_add_probe(
            osd_sink,
            GST_PAD_PROBE_TYPE_BUFFER,
            crosswalk_line_probe,
            nullptr,
            nullptr);

        gst_object_unref(osd_sink);
        std::cout << "[CROSSWALK] Line probe attached to OSD sink" << std::endl;
    }


    /* --- Detection probe --- */

    GstPad *osd_src =
        gst_element_get_static_pad(
            osd,
            "src");

    if (!osd_src)
    {
        std::cerr
            << "ERROR: Failed to get "
               "shared nvdsosd src pad."
            << std::endl;

        gst_object_unref(pipeline);
        g_main_loop_unref(main_loop);

        return 1;
    }


    gst_pad_add_probe(
        osd_src,
        GST_PAD_PROBE_TYPE_BUFFER,
        infer_src_probe,
        nullptr,
        nullptr);

    gst_object_unref(osd_src);


    /* --- appsink callback --- */

    g_signal_connect(
        GST_APP_SINK(appsink_element),
        "new-sample",
        G_CALLBACK(on_new_sample),
        nullptr);


    /* --- Bus --- */

    GstBus *bus =
        gst_element_get_bus(pipeline);

    gst_bus_add_watch(
        bus,
        bus_callback,
        main_loop);

    gst_object_unref(bus);


    /* --- Ctrl+C --- */

    g_unix_signal_add(
        SIGINT,
        stop_pipeline,
        main_loop);


    /* --- Print configuration --- */

    g_print(
        "\n========================================\n");

    g_print(
        " YOLO26 MERGED PIPELINE\n");

    g_print(
        " CAMERA JSON: %s\n",
        CAMERA_JSON_FILE);

    g_print(
        " RTSP INPUT:  %s\n",
        rtsp_url);

    g_print(
        " RTSP OUTPUT: %s\n",
        RTSP_OUTPUT_URL_HINT);

    g_print(
        " DeepStream config: %s\n",
        DEEPSTREAM_CONFIG);

    g_print(
        " Snapshot directory: %s\n",
        SNAPSHOT_DIR);

    g_print(
        " Event directory: %s\n",
        EVENT_DIR);

    g_print(
        " Vehicle snapshots: %s (one jpg per vehicle crossing the line)\n",
        VEHICLE_SNAPSHOT_DIR);

    g_print(
        " Branches: detection-only | "
        "snapshot+event | live RTSP out\n");

    g_print(
        "========================================\n\n");


    /* --- Start pipeline --- */

    GstStateChangeReturn state =
        gst_element_set_state(
            pipeline,
            GST_STATE_PLAYING);

    if (state == GST_STATE_CHANGE_FAILURE)
    {
        std::cerr
            << "ERROR: Failed to start pipeline."
            << std::endl;

        gst_element_set_state(
            pipeline,
            GST_STATE_NULL);

        gst_object_unref(pipeline);
        g_main_loop_unref(main_loop);

        return 1;
    }


    g_main_loop_run(main_loop);


    /* --- Cleanup --- */

    gst_element_set_state(
        pipeline,
        GST_STATE_NULL);

    gst_object_unref(pipeline);

    g_main_loop_unref(main_loop);

    std::cout
        << "Pipeline stopped."
        << std::endl;

    return 0;
}
