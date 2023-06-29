Shader "Unlit/NerfImitation"
{
    Properties
    {
        _MainTex ("Albedo (RGB)", 2D) = "white" {}
        _OctahedralMaps ("Octahedral Maps", 2D) = "white" {}
        _LabelMap ("Label Map", 2D) = "white" {}
        _LabelMapCountSqrt ("Label Map Sqrt", int) = 0
        _LabelMapCountSqrtRecip ("Label Map Sqrt Reciprocal", float) = 0 
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
                float3 worldPos : TEXCOORD1;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
            sampler2D _OctahedralMaps;
            float4 _OctahedralMaps_ST;
            float4 _OctahedralMaps_TexelSize;
            sampler2D _LabelMap;
            float4 _LabelMap_ST;

            uint _LabelMapCountSqrt;
            float _LabelMapCountSqrtRecip;

            v2f vert (appdata v)
            {
                float4 world_position = mul(unity_ObjectToWorld, v.vertex);
                float3 worldPos = world_position.xyz;
                
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                o.worldPos = worldPos;
                return o;
            }

            float2 octahedral_mapping(float3 co)
            {
                // projection onto octahedron
                co /= dot( float3(1, 1, 1), abs(co) );

                // out-folding of the downward faces
                float2 uv = float2(0.0f, 0.0f);
                if ( co.y < -0.000001f ) {
                    uv.y = (1.0f - abs(co.x)) * sign(co.z);
                    uv.x = (1.0f - abs(co.z)) * sign(co.x);
                } else {
                    uv.x = co.x;
                    uv.y = co.z;
                }

                // mapping to [0;1]ˆ2 texture space
                return uv * 0.5f + 0.5f;
            }
            
            fixed3 evaluateNetwork(float2 vUv, float3 viewdir) {
                // Width, height, (depth)

                float4 label = tex2D(_LabelMap, vUv);

                float2 octaUV = octahedral_mapping(viewdir);
                
                int iLabel = int(round(label.x * 255.0f));

                float iLabelY = float(iLabel % _LabelMapCountSqrt);
                float iLabelX = float(iLabel / _LabelMapCountSqrt);
                
                float oneOcta = float(_OctahedralMaps_TexelSize.z / _LabelMapCountSqrt);

                float mappingX = (iLabelX * oneOcta + octaUV.x * oneOcta) * _OctahedralMaps_TexelSize.x;
                float mappingY = (iLabelY * oneOcta + octaUV.y * oneOcta) * _OctahedralMaps_TexelSize.y;

                float4 finalTexel = tex2D(_OctahedralMaps, float2(mappingX, mappingY));

                return finalTexel.xyz;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                fixed4 diffuse = tex2D(_MainTex, i.uv);
                // diffuse = fixed4(0,0,0,0);
                float3 rayDirection = i.worldPos.xyz - _WorldSpaceCameraPos;
                rayDirection = float3(-rayDirection.x, -rayDirection.z, rayDirection.y);
                fixed4 col = fixed4(clamp(diffuse.rgb + evaluateNetwork(i.uv, normalize(rayDirection)), 0.0f, 1.0f), 1);
                return col;
            }
            ENDCG
        }
    }
}
