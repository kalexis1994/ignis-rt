// hair_cylinder.glsl — Hair cylinder intersection + cylindrical normal computation
// Shared between monolithic raygen and wavefront wf_shade.comp
// Requires: Vertices, Indices, Normals buffer references, GeometryMetadata, objToWorld

#ifndef HAIR_CYLINDER_GLSL
#define HAIR_CYLINDER_GLSL

const uint INVALID_HAIR_STRAND = 0xFFFFFFFFu;

vec3 octDecode_hair(vec2 e) {
    e = e * 2.0 - 1.0;
    vec3 n = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    }
    return normalize(n);
}

bool getHairRibbonLayout(GeometryMetadata geo,
                         out uint strandCount,
                         out uint vertsPerStrand,
                         out uint trisPerStrand)
{
    strandCount = 0u;
    vertsPerStrand = 0u;
    trisPerStrand = 0u;

    if (geo.bitangentBufferAddress == 0u || geo.indexCount < 12u || geo.vertexCount < 8u)
        return false;

    uint totalTris = geo.indexCount / 3u;
    if (geo.vertexCount <= totalTris)
        return false;

    uint vertexMinusTris = geo.vertexCount - totalTris;
    if ((vertexMinusTris & 3u) != 0u)
        return false;

    strandCount = vertexMinusTris / 4u;
    if (strandCount == 0u || (totalTris % strandCount) != 0u)
        return false;

    trisPerStrand = totalTris / strandCount;
    if ((trisPerStrand & 3u) != 0u)
        return false;

    vertsPerStrand = trisPerStrand + 4u;
    if (geo.vertexCount != strandCount * vertsPerStrand)
        return false;

    return true;
}

vec3 fetchHairVertexWorld(Vertices vertices, mat4x3 objToWorld, uint vertexIndex)
{
    return (objToWorld * vec4(
        vec3(vertices.v[vertexIndex*3u + 0u],
             vertices.v[vertexIndex*3u + 1u],
             vertices.v[vertexIndex*3u + 2u]), 1.0));
}

// Full hair cylinder intersection: computes cylindrical normal, hairV, shading position
// Outputs are written to the out parameters
void computeHairCylinder(
    GeometryMetadata geo, Vertices vertices, Indices indices, Normals normals,
    mat4x3 objToWorld, vec3 rayOrigin, vec3 rayDir, vec3 worldPos,
    uint i0, uint i1, uint i2, vec3 wp0, vec3 wp1, vec3 wp2,
    vec3 bary, int primitiveId,
    out vec3 outWorldN, out vec3 outSmoothT, out float outHairV,
    out vec3 outShadePos, out uint outStrandId)
{
    // Decode packed tangent from normals (oct-encoded in yz components)
    vec3 rawN = vec3(normals.n[i0*3+0], normals.n[i0*3+1], normals.n[i0*3+2]) * bary.x
              + vec3(normals.n[i1*3+0], normals.n[i1*3+1], normals.n[i1*3+2]) * bary.y
              + vec3(normals.n[i2*3+0], normals.n[i2*3+1], normals.n[i2*3+2]) * bary.z;
    vec3 localTangent = octDecode_hair(vec2(rawN.y, rawN.z));
    outSmoothT = normalize(mat3(objToWorld) * localTangent);
    outStrandId = INVALID_HAIR_STRAND;
    outHairV = 0.0;
    outShadePos = worldPos;

    // Find center line of the ribbon from same-cross-section left/right pair
    float nv0 = normals.n[i0*3];
    float nv1 = normals.n[i1*3];
    float nv2 = normals.n[i2*3];
    vec3 centerPt;
    vec3 widthEdge;
    vec3 otherWidthEdge;
    bool found = false;
    if (!found && nv0 * nv2 < 0.0 && abs(int(i0) - int(i2)) == 1) {
        centerPt = (wp0 + wp2) * 0.5; widthEdge = wp2 - wp0; found = true;
    }
    if (!found && nv0 * nv1 < 0.0 && abs(int(i0) - int(i1)) == 1) {
        centerPt = (wp0 + wp1) * 0.5; widthEdge = wp1 - wp0; found = true;
    }
    if (!found && nv1 * nv2 < 0.0 && abs(int(i1) - int(i2)) == 1) {
        centerPt = (wp1 + wp2) * 0.5; widthEdge = wp2 - wp1; found = true;
    }
    if (!found) {
        centerPt = (wp0 + wp1 + wp2) / 3.0;
        widthEdge = wp2 - wp0;
    }

    // Adjacent triangle for second center point
    uint adjPrimId = (uint(primitiveId) % 2u == 0u) ? uint(primitiveId) + 1u : uint(primitiveId) - 1u;
    uint adj_i0 = indices.i[adjPrimId * 3u + 0u];
    uint adj_i1 = indices.i[adjPrimId * 3u + 1u];
    uint adj_i2 = indices.i[adjPrimId * 3u + 2u];
    vec3 adj_wp0 = fetchHairVertexWorld(vertices, objToWorld, adj_i0);
    vec3 adj_wp1 = fetchHairVertexWorld(vertices, objToWorld, adj_i1);
    vec3 adj_wp2 = fetchHairVertexWorld(vertices, objToWorld, adj_i2);

    float adj_nv0 = normals.n[adj_i0*3u];
    float adj_nv1 = normals.n[adj_i1*3u];
    float adj_nv2 = normals.n[adj_i2*3u];
    vec3 otherCenter = (adj_wp0 + adj_wp1 + adj_wp2) / 3.0;
    otherWidthEdge = adj_wp2 - adj_wp0;
    if (adj_nv0 * adj_nv2 < 0.0 && abs(int(adj_i0)-int(adj_i2)) == 1) {
        otherCenter = (adj_wp0 + adj_wp2) * 0.5; otherWidthEdge = adj_wp2 - adj_wp0;
    } else if (adj_nv0 * adj_nv1 < 0.0 && abs(int(adj_i0)-int(adj_i1)) == 1) {
        otherCenter = (adj_wp0 + adj_wp1) * 0.5; otherWidthEdge = adj_wp1 - adj_wp0;
    } else if (adj_nv1 * adj_nv2 < 0.0 && abs(int(adj_i1)-int(adj_i2)) == 1) {
        otherCenter = (adj_wp1 + adj_wp2) * 0.5; otherWidthEdge = adj_wp2 - adj_wp1;
    }

    // Hair section layout for precise cylinder reconstruction
    bool haveHairSections = false;
    vec3 sec0a, sec0b, sec0c, sec0d, sec1a, sec1b, sec1c, sec1d;
    vec3 prevCenter = centerPt, nextCenter = otherCenter;
    bool havePrevCenter = false, haveNextCenter = false;
    uint strandCount, vertsPerStrand, trisPerStrand;
    if (getHairRibbonLayout(geo, strandCount, vertsPerStrand, trisPerStrand)) {
        outStrandId = uint(primitiveId) / trisPerStrand;
        uint localPrimId = uint(primitiveId) % trisPerStrand;
        uint segId = localPrimId / 4u;
        uint strandBase = outStrandId * vertsPerStrand;
        uint strandEnd = strandBase + vertsPerStrand;
        uint section0Base = outStrandId * vertsPerStrand + segId * 4u;
        uint section1Base = section0Base + 4u;
        if (section1Base + 3u < geo.vertexCount) {
            sec0a = fetchHairVertexWorld(vertices, objToWorld, section0Base + 0u);
            sec0b = fetchHairVertexWorld(vertices, objToWorld, section0Base + 1u);
            sec0c = fetchHairVertexWorld(vertices, objToWorld, section0Base + 2u);
            sec0d = fetchHairVertexWorld(vertices, objToWorld, section0Base + 3u);
            sec1a = fetchHairVertexWorld(vertices, objToWorld, section1Base + 0u);
            sec1b = fetchHairVertexWorld(vertices, objToWorld, section1Base + 1u);
            sec1c = fetchHairVertexWorld(vertices, objToWorld, section1Base + 2u);
            sec1d = fetchHairVertexWorld(vertices, objToWorld, section1Base + 3u);
            centerPt = (sec0a + sec0b + sec0c + sec0d) * 0.25;
            otherCenter = (sec1a + sec1b + sec1c + sec1d) * 0.25;
            widthEdge = (sec0b - sec0a) + (sec0d - sec0c);
            otherWidthEdge = (sec1b - sec1a) + (sec1d - sec1c);
            haveHairSections = true;

            if (segId > 0u && section0Base >= strandBase + 4u) {
                uint prevBase = section0Base - 4u;
                prevCenter = (fetchHairVertexWorld(vertices, objToWorld, prevBase + 0u)
                            + fetchHairVertexWorld(vertices, objToWorld, prevBase + 1u)
                            + fetchHairVertexWorld(vertices, objToWorld, prevBase + 2u)
                            + fetchHairVertexWorld(vertices, objToWorld, prevBase + 3u)) * 0.25;
                havePrevCenter = true;
            }
            if (section1Base + 7u < strandEnd) {
                uint nextBase = section1Base + 4u;
                nextCenter = (fetchHairVertexWorld(vertices, objToWorld, nextBase + 0u)
                            + fetchHairVertexWorld(vertices, objToWorld, nextBase + 1u)
                            + fetchHairVertexWorld(vertices, objToWorld, nextBase + 2u)
                            + fetchHairVertexWorld(vertices, objToWorld, nextBase + 3u)) * 0.25;
                haveNextCenter = true;
            }
        }
    }

    // Chord axis and smooth tangent
    vec3 chordAxis = otherCenter - centerPt;
    float chordLen = length(chordAxis);
    vec3 chordT = (chordLen > 1e-8) ? chordAxis / chordLen : outSmoothT;
    vec3 hairGeomT = chordT;
    if (havePrevCenter && haveNextCenter) {
        vec3 smoothAxis = nextCenter - prevCenter;
        float smoothLen = length(smoothAxis);
        if (smoothLen > 1e-8) hairGeomT = smoothAxis / smoothLen;
    } else if (havePrevCenter) {
        vec3 prevAxis = otherCenter - prevCenter;
        if (length(prevAxis) > 1e-8) hairGeomT = normalize(prevAxis);
    } else if (haveNextCenter) {
        vec3 nextAxis = nextCenter - centerPt;
        if (length(nextAxis) > 1e-8) hairGeomT = normalize(nextAxis);
    }
    if (length(hairGeomT) < 1e-6) hairGeomT = outSmoothT;

    // Cylinder radius from cross-section geometry
    float radius0, radius1;
    if (haveHairSections) {
        vec3 sec0o0 = sec0a - centerPt - hairGeomT * dot(sec0a - centerPt, hairGeomT);
        vec3 sec0o1 = sec0b - centerPt - hairGeomT * dot(sec0b - centerPt, hairGeomT);
        vec3 sec0o2 = sec0c - centerPt - hairGeomT * dot(sec0c - centerPt, hairGeomT);
        vec3 sec0o3 = sec0d - centerPt - hairGeomT * dot(sec0d - centerPt, hairGeomT);
        vec3 sec1o0 = sec1a - otherCenter - hairGeomT * dot(sec1a - otherCenter, hairGeomT);
        vec3 sec1o1 = sec1b - otherCenter - hairGeomT * dot(sec1b - otherCenter, hairGeomT);
        vec3 sec1o2 = sec1c - otherCenter - hairGeomT * dot(sec1c - otherCenter, hairGeomT);
        vec3 sec1o3 = sec1d - otherCenter - hairGeomT * dot(sec1d - otherCenter, hairGeomT);
        radius0 = max((length(sec0o0) + length(sec0o1) + length(sec0o2) + length(sec0o3)) * 0.25, 1e-6);
        radius1 = max((length(sec1o0) + length(sec1o1) + length(sec1o2) + length(sec1o3)) * 0.25, 1e-6);
        if (length(widthEdge) < 1e-6) widthEdge = sec0b - sec0a;
    } else {
        radius0 = max(length(widthEdge) * 0.5, 1e-6);
        radius1 = max(length(otherWidthEdge) * 0.5, 1e-6);
    }
    float widthLen = length(widthEdge);
    vec3 wDir = (widthLen > 1e-6) ? widthEdge / widthLen : vec3(1, 0, 0);

    // Segment interpolation with smoothstep for C1 continuity
    float segT = clamp(dot(worldPos - centerPt, chordT) / max(chordLen, 1e-8), 0.0, 1.0);
    segT = smoothstep(0.0, 1.0, segT);

    vec3 sweepCenter = mix(centerPt, otherCenter, segT);
    vec3 toHit = worldPos - sweepCenter;
    vec3 perpToAxis = toHit - hairGeomT * dot(toHit, hairGeomT);

    float coneRadius = mix(radius0, radius1, segT);
    vec3 camLR = cross(hairGeomT, -rayDir);
    float lrLen = length(camLR);
    if (lrLen > 1e-6) camLR /= lrLen; else camLR = wDir;

    // Ray-cylinder intersection: virtual cylinder normal
    vec3 relOrigin = rayOrigin - sweepCenter;
    vec3 D_perp = rayDir - hairGeomT * dot(rayDir, hairGeomT);
    vec3 O_perp = relOrigin - hairGeomT * dot(relOrigin, hairGeomT);

    float rcA = dot(D_perp, D_perp);
    float rcB = 2.0 * dot(O_perp, D_perp);
    float rcC = dot(O_perp, O_perp) - coneRadius * coneRadius;
    float disc = rcB * rcB - 4.0 * rcA * rcC;

    vec3 cylN;
    if (disc >= 0.0 && rcA > 1e-8) {
        float sqrtDisc = sqrt(max(disc, 0.0));
        float t0 = (-rcB - sqrtDisc) / (2.0 * rcA);
        float t1 = (-rcB + sqrtDisc) / (2.0 * rcA);
        float t_cyl = (t0 > 1e-5) ? t0 : ((t1 > 1e-5) ? t1 : -1.0);

        if (t_cyl > 0.0) {
            outShadePos = rayOrigin + t_cyl * rayDir;
            vec3 cylEntry = outShadePos - sweepCenter;
            cylEntry -= hairGeomT * dot(cylEntry, hairGeomT);
            float entryLen = length(cylEntry);
            if (entryLen > 1e-6) {
                cylN = cylEntry / entryLen;
            } else {
                cylN = (length(perpToAxis) > 1e-6) ? normalize(perpToAxis) : wDir;
                outShadePos = sweepCenter + cylN * coneRadius;
            }
            outHairV = clamp(dot(cylN * coneRadius, camLR) / max(coneRadius, 1e-8), -1.0, 1.0);
        } else {
            cylN = (length(perpToAxis) > 1e-6) ? normalize(perpToAxis) : wDir;
            outShadePos = sweepCenter + cylN * coneRadius;
            outHairV = clamp(dot(cylN * coneRadius, camLR) / max(coneRadius, 1e-8), -1.0, 1.0);
        }
    } else {
        cylN = (length(perpToAxis) > 1e-6) ? normalize(perpToAxis) : wDir;
        outShadePos = sweepCenter + cylN * coneRadius;
        outHairV = clamp(dot(cylN * coneRadius, camLR) / max(coneRadius, 1e-8), -1.0, 1.0);
    }

    outWorldN = cylN;
}

#endif // HAIR_CYLINDER_GLSL
