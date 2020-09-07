//
// Created by jglrxavpok on 07/09/2020.
//

#include "TriangleMesh.h"
#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include <vector>
#include "cudautils.h"
#include "Material.h"
#include "materials/Metal.h"

__device__ TriangleMesh::TriangleMesh(IntersectableGroup* group): backingRepresentation(group) {}

template<typename T, typename ... Args>
__global__
void allocate(T** destination, Args ... args) {
    *destination = new T(args...);
}

template<typename T, typename ... Args>
T* deviceNew(Args ... args) {
    T* result;
    cudaMallocManaged(&result, sizeof(T*));
    allocate<T> <<<1, 1>>>(&result, args...);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaPeekAtLastError());
    return result;
}

IntersectableGroup* processMesh(const aiMesh* mesh, const aiScene* scene) {
    const bool hasNormals = mesh->mNormals != nullptr;
    const bool hasTexture = mesh->mTextureCoords[0] != nullptr;
    std::vector<Triangle*> triangles;
    std::vector<Point3> vertices;
    std::vector<Vec3> normals;
    std::vector<Point3> texCoords;

    // FIXME: use material from file
    Material* material = deviceNew<Metal>(Color(1,1,1), 0.1);

    for(int i = 0; i < mesh->mNumVertices; i++) {
        const auto v = mesh->mVertices[i];
        vertices.emplace_back(v.x, v.y, v.z);
        if(hasNormals) {
            auto normal = mesh->mNormals[i];
            normals.emplace_back(normal.x, normal.y, normal.z);
        } else {
            normals.emplace_back(0,0,0);
        }

        if(hasTexture) {
            auto uv = mesh->mTextureCoords[0][i];
            texCoords.emplace_back(uv.x, uv.y, 0.0);
        } else {
            texCoords.emplace_back(0.0, 0.0, 0.0);
        }
    }

    for(int i = 0; i < mesh->mNumFaces; i++) {
        auto face = mesh->mFaces[i];
        if(face.mNumIndices != 3) {
            std::cerr << "Cannot load model that is not triangulated" << std::endl;
            exit(-1);
        }

        auto pos0 = vertices[face.mIndices[0]];
        auto pos1 = vertices[face.mIndices[1]];
        auto pos2 = vertices[face.mIndices[2]];
        auto normal0 = normals[face.mIndices[0]];
        auto normal1 = normals[face.mIndices[1]];
        auto normal2 = normals[face.mIndices[2]];
        auto uvw0 = texCoords[face.mIndices[0]];
        auto uvw1 = texCoords[face.mIndices[1]];
        auto uvw2 = texCoords[face.mIndices[2]];
        triangles.push_back(deviceNew<Triangle>(pos0, pos1, pos2, normal0, normal1, normal2, uvw0, uvw1, uvw2, material));
    }

    Intersectable** components;
    checkCudaErrors(cudaMallocManaged(&components, triangles.size()*sizeof(Intersectable*)));
    for(int i = 0; i < triangles.size(); i++) {
        components[i] = triangles[i];
    }

    return deviceNew<IntersectableGroup>(triangles.size(), components);
}

IntersectableGroup* processNodes(const aiNode* node, const aiScene* scene) {
    std::vector<IntersectableGroup*> meshes{};
    for(int i = 0; i < node->mNumMeshes; i++) {
        meshes.push_back(processMesh(scene->mMeshes[node->mMeshes[i]], scene));
    }

    for(int i = 0; i < node->mNumChildren; i++) {
        meshes.push_back(processNodes(node->mChildren[i], scene));
    }

    Intersectable** components;
    checkCudaErrors(cudaMallocManaged(&components, meshes.size()*sizeof(Intersectable*)));
    for(int i = 0; i < meshes.size(); i++) {
        components[i] = meshes[i];
    }

    return deviceNew<IntersectableGroup>(meshes.size(), components);
}

__host__ TriangleMesh* TriangleMesh::loadFromFile(const std::string &name) {
    std::cout << "Loading " << name << std::endl;

    Assimp::Importer importer{};
    const aiScene* scene = importer.ReadFile(name, aiProcess_Triangulate | aiProcess_GenNormals);
    if(!scene) {
        std::cerr << "Failed to load mesh " << name << std::endl;
        exit(-1);
    }

    IntersectableGroup* group = processNodes(scene->mRootNode, scene);

    // TODO: materials
    return deviceNew<TriangleMesh>(group);
}

__device__ bool TriangleMesh::hit(const Ray &ray, double mint, double maxt, HitResult &result) const {
    return backingRepresentation->hit(ray, mint, maxt, result);
}
