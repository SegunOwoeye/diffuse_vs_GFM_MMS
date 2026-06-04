#pragma once

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace core {

enum class ModelType {
    Auto,
    SingleMaterialEuler,
    DiffuseInterface,
    SharpInterface,
    ElasticSolid,
    ElastoplasticSolid
};

inline std::string normalise_model_type(std::string value)
{
    const auto first = value.find_first_not_of(" \t\r\n");
    const auto last = value.find_last_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    value = value.substr(first, last - first + 1);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        if (c == '-' || c == ' ') {
            return '_';
        }
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

inline ModelType parse_model_type(const std::string& raw)
{
    const std::string value = normalise_model_type(raw);
    if (value.empty() || value == "auto") {
        return ModelType::Auto;
    }
    if (value == "sm" || value == "single_material" ||
        value == "single_material_euler" || value == "euler") {
        return ModelType::SingleMaterialEuler;
    }
    if (value == "dim" || value == "diffuse_interface" ||
        value == "diffuse_interface_model") {
        return ModelType::DiffuseInterface;
    }
    if (value == "sim" || value == "gfm" || value == "rgfm" ||
        value == "sharp_interface" || value == "ghost_fluid") {
        return ModelType::SharpInterface;
    }
    if (value == "elastic_solid" || value == "solid_elastic") {
        return ModelType::ElasticSolid;
    }
    if (value == "elastoplastic_solid" || value == "solid_elastoplastic" ||
        value == "elasto_plastic_solid" || value == "plastic_solid") {
        return ModelType::ElastoplasticSolid;
    }
    throw std::runtime_error("Unknown model_type: " + raw);
}

inline const char* model_type_name(ModelType type)
{
    switch (type) {
        case ModelType::Auto: return "auto";
        case ModelType::SingleMaterialEuler: return "single_material_euler";
        case ModelType::DiffuseInterface: return "diffuse_interface";
        case ModelType::SharpInterface: return "sharp_interface";
        case ModelType::ElasticSolid: return "elastic_solid";
        case ModelType::ElastoplasticSolid: return "elastoplastic_solid";
    }
    return "auto";
}

inline ModelType model_type_from_interface_method(const std::string& method)
{
    const std::string value = normalise_model_type(method);
    if (value == "sm") {
        return ModelType::SingleMaterialEuler;
    }
    if (value == "dim") {
        return ModelType::DiffuseInterface;
    }
    if (value == "gfm" || value == "sim" || value == "rgfm") {
        return ModelType::SharpInterface;
    }
    throw std::runtime_error("Cannot infer model_type from interface_method: " + method);
}

inline bool model_type_matches_interface_method(ModelType type, const std::string& method)
{
    if (type == ModelType::Auto) {
        return true;
    }
    return type == model_type_from_interface_method(method);
}

inline bool is_solid_model_type(ModelType type)
{
    return type == ModelType::ElasticSolid || type == ModelType::ElastoplasticSolid;
}

} // namespace core
