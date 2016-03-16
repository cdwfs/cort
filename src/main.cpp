#include "cds_vector.h"

#include <float.h>
#include <stdio.h>

bool intersectRayBox(float3 rayOrg, float3 invDir, float3 bbmin, float3 bbmax, float &hitT)
{
    float3 d0 = (bbmin - rayOrg) * invDir;
    float3 d1 = (bbmax - rayOrg) * invDir;

    float3 v0 = min(d0, d1);
    float3 v1 = max(d0, d1);

    float tmin = hmax(v0);
    float tmax = hmin(v1);

    bool hit = (tmax >= 0) && (tmax >= tmin) && (tmin <= hitT);
    if (hit)
        hitT = tmin;
    return hit;
}

int __cdecl main(int argc, char *argv[])
{
	float3 testorg(0,0,0);
	float3 testdir = float3(3,2,1);
	float3 testbbmin = float3(10,0,0);
	float3 testbbmax = float3(20,10,10);

	float hitT = FLT_MAX;
	bool hit = intersectRayBox(testorg, float3(1, 1, 1) / testdir, testbbmin, testbbmax, hitT);
	printf("hit %i at t=%f\n", hit, hitT);

	return 0;
}
