using UnityEngine;

public class Assignment1 : MonoBehaviour
{

    int[][] D = { new int[]{0, 0}, new int[] {3, 4}, new int[] {5, 5}, new int[] {10, 10}, new int[] {100, 100} };

    int[] y = {7, 7};

    private void Start()
    {
        var a = GetSmallestDistance(D, y);

        if (a != null)
        {
            string s = "The Smallest Distance Vector is {";
            for (int i = 0; i < a.Length; i++)
            {
                s += a[i].ToString();
                if (i < a.Length - 1) s += ", ";
            }
            s += "}";

            Debug.Log(s);
        }
    }

    public int[] GetSmallestDistance(int[][] D, int[] vector)
    {
        if(D[0].Length != vector.Length)
        {
            Debug.LogError("# Set D and given vector are not in same dimension");
            return null;
        }

        float minDistance = int.MaxValue;
        int minPointer = 0;

        for(int i = 0; i < D.Length; i++)
        {
            float distnace = EuclideanDistance(D[i], vector);

            if (distnace < minDistance)
            {
                minDistance = distnace;
                minPointer = i;
            }
        }

        return D[minPointer];
    }

    private float EuclideanDistance(int[] v1, int[] v2)
    {
        if(v1.Length != v2.Length)
        {
            Debug.LogError("# Those two vector are not in same dimension");
            return float.MaxValue; // Maybe there's better way for exception
        }

        int dimension = v1.Length;
        float returnValue = 0;

        for(int i = 0; i < dimension; i++)
        {
            returnValue += (v1[i] - v2[i]) * (v1[i] - v2[i]);
        }

        returnValue = Mathf.Sqrt(returnValue);

        return returnValue;
    }

}
