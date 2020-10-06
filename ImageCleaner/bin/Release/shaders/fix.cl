int orientare(float2 A, float2 B, float2 C) {
	float d = (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);
	if (d > 0) return 1;
  else if (d < 0) return -1;
	return 0;
}

uint findprojectionind(int k, float2 O, float2 B, __local float2* Q)
{ 
  int j1 = 0;
  int m = 0;

  for(int i = 0; i< 4; i++) {
    if((orientare(B, Q[i], Q[i+1]) * orientare(O, Q[i], Q[i+1])) < 0) {
      j1++;
      m = j1;
    }
  }

  return (uint)m;
}

int intersect(float2 A, float2 B, float2 C, float2 D) {
  return ((orientare(A, B, C) * orientare(A, B, D) < 0) && (orientare(C, D, A) * orientare(C, D, B) < 0));
}

int point_inside(float2 point, int len, __local float2* Q) {
    float2 a = Q[0];
    int count = 0;

    float2 R = (float2)(point.x + 1000, point.y);

    for(int i = 0; i < len; i++) {
        if (intersect(point, R, Q[i], Q[i+1])) 
          count++;
    }

    return (count & 0x1);
  }

float2 project(float2 M, float2 A, float2 B) {
      float2 S = (float2)0;
      float2 e1 = B - A; 
      float2 e2 = M - A;

      float l1 = length(e1);
      float l2 = length(e2);
      
      float lproj = l2 * dot(e1, e2) / (l1 * l2);
      S.x = A.x + ((lproj / l1) * e1.x);
      S.y = A.y + ((lproj / l1) * e1.y);
      return S;
}

float2 ca(int len, __local float2* A, __local float* w, float coef, float2 B) {
  float2 C = (float2)0;
  float2 C1 = (float2)0;
   
  for(int i = 0; i < len; i++) {
    C.x += (w[i] * A[i].x);
    C.y += (w[i] * A[i].y);
  }

  C1.x = B.x + (coef * (C.x - B.x));
  C1.y = B.y + (coef * (C.y - B.y));
  return C1;
}

__kernel void fixImage(__global float4* dest, __global float2* P, __global float2* data, __global float* ptr, int len) {
  __local float2 A[128];
  __local float2 Q[128];
  __local float2 R[128];

  __local float w[128];
  __local float lambda[128];
  __local float d[128];

  for(int j = 0; j < 128; j++) lambda[j] = 1 + (1.0f / (j + 1.0f)); //1.7f;
  float2 O = (float2)(ptr[1], ptr[2]);
  float rmax = ptr[0];

  int j = get_global_id(0) * 10240 + get_global_id(1);
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  
  for(int i = 0; i < (len >> 1) - 1; i++) {
    R[0] = P[j];
    Q[0] = data[i];

    Q[1] = data[i + 1];
    Q[2] = data[len - i - 2];
    Q[3] = data[len - i - 1];
    Q[4] = Q[0];
  
    int n = 4;
    int ins = !point_inside(R[0], len, Q);
    float sdp = 0.0f, sdp0 = 0.0f;
    int k = 0;

    if(ins) {
      do { 
        int m = findprojectionind(k, O, R[k], Q);
  
        for (int j1 = 0; j1 < n; j1++)  
            A[j1] = project(R[k], Q[j1], Q[j1 + 1]);

        m = n;

      if(m > 0) { 
          for (int i1 = 0; i1 < m; i1++) d[i1] = distance(R[k], A[i1]);
        float sumd = 0;
        float sdp = 0;
        for (int i1 = 0; i1 < m; i1++) sumd += (1.0f / (d[i1] + 1));
        for (int i1 = 0; i1 < m; i1++) sdp += (d[i1] * d[i1]);
        if(k == 1) sdp0 = sdp;
        for (int i1 = 0; i1 < m; i1++) w[i1] = (1.0f / (d[i1] + 1)) / sumd; 
      }  
    else {
      w[0] = 1;
      sdp = 0;
    }
     k++;
     barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
     R[k] = ca(m, A, w, lambda[k-1], R[k-1]);
     ins = !point_inside(R[k], len, Q);
    } while(ins && k < 45);
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      dest[j] = (float4)(R[k], R[k - 1]);
    }
  }
}
