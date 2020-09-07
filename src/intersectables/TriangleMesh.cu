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

__global__
void allocateMaterial(Material** destination) {
    *destination = new Metal(Color(1,1,1), 0.1);
}

__global__
void allocateTriangle(Triangle** destination, Point3 a, Point3 b, Point3 c, Vec3 normalA, Vec3 normalB, Vec3 normalC,
        Vec3 uvA, Vec3 uvB, Vec3 uvC, Material* material) {
    *destination = new Triangle(a, b, c, normalA, normalB, normalC, uvA, uvB, uvC, material);
}

template<typename T>
T* newTriangle(Point3 a, Point3 b, Point3 c, Vec3 normalA, Vec3 normalB, Vec3 normalC,
                 Vec3 uvA, Vec3 uvB, Vec3 uvC, Material* material) {
    T* result;
    cudaMallocManaged(&result, sizeof(T*));

    // TODO
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaPeekAtLastError());
    return result;
}

IntersectableGroup* processMesh(const aiMesh* mesh, const aiScene* scene) {
    const bool hasNormals = mesh->mNormals != nullptr;
    std::vector<Triangle*> triangles;

    for(int i = 0; i < mesh->mNumVertices; i++) {
        const auto v = mesh->mVertices[i];

    }
// TODO
    return nullptr;
}

void processNodes(const aiNode* node, const aiScene* scene) {
    std::vector<IntersectableGroup*> meshes{};
    for(int i = 0; i < node->mNumMeshes; i++) {
        meshes.push_back(processMesh(scene->mMeshes[node->mMeshes[i]], scene));
    }

    for(int i = 0; i < node->mNumChildren; i++) {
        processNodes(node->mChildren[i], scene);
    }
}

__host__ TriangleMesh **TriangleMesh::loadFromFile(const std::string &name) {
    Assimp::Importer importer{};
    const aiScene* scene = importer.ReadFile(name, aiProcess_Triangulate | aiProcess_GenNormals);
    if(!scene) {
        std::cerr << "Failed to load mesh " << name << std::endl;
        exit(-1);
    }

    processNodes(scene->mRootNode, scene);

    // TODO: materials
    return nullptr;
}
