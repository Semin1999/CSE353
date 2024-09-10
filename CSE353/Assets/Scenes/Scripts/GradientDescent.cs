using UnityEngine;

public class GradientDescent : MonoBehaviour
{
    private float next_x = 6; // start point at x = 6
    private float gamma = 0.01f; // step size multiplier
    private float pricision = 0.00001f; // Desired precision of result
    private int MAX_ITER = 100000;

    private void Start()
    {
        GetOptimization(MAX_ITER);
    }

    private float DifferentiateFunction(float x)
    {
        return 4 * x * x * x - 9 * x * x;
    }

    public void GetOptimization(int MAX_ITER)
    {
        for(int i = 0; i < MAX_ITER; i++)
        {
            float current_x = next_x;

            next_x = current_x - gamma * DifferentiateFunction(current_x);

            float step = next_x - current_x;

            Debug.Log($"currentX = {current_x} / nextX = {next_x} / step = {step} / gradient = {DifferentiateFunction(current_x)}");

            if (Mathf.Abs(step) <= pricision)
                break;
        }

        print($"## -- MINIMUM AT :{next_x}");
    }
}
