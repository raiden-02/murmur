#include "lab/lab.hpp"

#include "engine/engine.hpp"
#include "engine/model_loader.hpp"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>
#include <map>
#include <optional>
#include <print>
#include <string>
#include <vector>

namespace murmur::lab {

namespace {

struct Arrival {
    int step;
    ConversationId conversation;
    std::string message;
};

}  // namespace

// Continuous batching, step by step: five conversations arrive at different
// steps (one of them the player's, at higher priority) with a small prompt
// budget per step, and every step's batch is printed: a number is that many
// prompt tokens, D a decode token, a dot nothing. Greedy, so the run is
// repeatable.
int RunScheduleTrace(Args args) {
    std::string json_path;
    ArgParser parser("murmur-lab schedule");
    parser.Text("--json", "<path>", json_path, "also write the trace there, for docs/diagrams/schedule_trace.svg");
    if (const std::optional<int> exit_code = parser.Parse(args)) {
        return *exit_code;
    }
    LoadedModel model = LoadModel(DefaultModelDirectory());
    Engine engine(*model.transformer, model.tokenizer, {.max_prefill_tokens = 24});

    const std::vector<std::pair<std::string, int>> cast = {
        {"Mei", 0}, {"Kenji", 0}, {"Aiko", 0}, {"Rafael", 0}, {"you", 1}};
    for (std::size_t id = 0; id < cast.size(); ++id) {
        ConversationOptions options;
        options.system_prompt = "You are " + cast[id].first + ", a vendor in a neon-lit night market. "
                                "Answer in one short sentence.";
        options.sampling = {.temperature = 0.0f, .top_k = 0, .top_p = 1.0f, .repetition_penalty = 1.0f};
        options.priority = cast[id].second;
        options.max_reply_tokens = 10;
        engine.Open(static_cast<ConversationId>(id), options);
    }
    const std::vector<Arrival> arrivals = {
        {0, 0, "What's good tonight?"},
        {0, 1, "Can you fix my phone?"},
        {3, 2, "Do you sell charms against drones?"},
        {5, 3, "How much for a skewer, and what is in the sauce?"},
        {6, 4, "Hi! Who runs this market?"},
    };

    nlohmann::json trace = {{"names", nlohmann::json::array()}, {"priority", nlohmann::json::array()},
                            {"steps", nlohmann::json::array()}};
    for (const auto& [name, priority] : cast) {
        trace["names"].push_back(name);
        trace["priority"].push_back(priority);
    }
    std::print("{:>5}", "step");
    for (const auto& [name, priority] : cast) {
        std::print(" {:>7}", name);
    }
    std::println("");
    std::size_t next_arrival = 0;
    for (int step = 0; step < 200; ++step) {
        while (next_arrival < arrivals.size() && arrivals[next_arrival].step == step) {
            engine.Send(arrivals[next_arrival].conversation, arrivals[next_arrival].message);
            ++next_arrival;
        }
        if (!engine.HasWork() && next_arrival == arrivals.size()) {
            break;
        }
        std::map<ConversationId, bool> finished;
        for (const EngineEvent& event : engine.Step()) {
            if (event.kind == EngineEventKind::kReplyFinished) {
                finished[event.conversation] = true;
            }
        }
        const EngineStats stats = engine.Stats();
        nlohmann::json row = nlohmann::json::array();
        std::print("{:>5}", step);
        // Per conversation: the tokens it ran this step (one is a decode token).
        for (std::size_t id = 0; id < cast.size(); ++id) {
            int tokens = 0;
            for (std::size_t slot = 0; slot < stats.last_batch.size(); ++slot) {
                if (stats.last_batch[slot] == static_cast<ConversationId>(id)) {
                    tokens = stats.last_batch_tokens[slot];
                }
            }
            row.push_back(tokens);
            std::print(" {:>7}", tokens == 0 ? std::string(".") : tokens == 1 ? std::string("D") : std::to_string(tokens));
        }
        nlohmann::json done = nlohmann::json::array();
        for (const auto& [id, yes] : finished) {
            done.push_back(id);
        }
        trace["steps"].push_back({{"tokens", row}, {"finished", done}});
        std::println("   prompt rows {}", stats.last_prefill_rows);
    }
    if (!json_path.empty()) {
        std::ofstream(json_path) << trace.dump(1);
        std::println("wrote {}", json_path);
    }
    return EXIT_SUCCESS;
}

}  // namespace murmur::lab
