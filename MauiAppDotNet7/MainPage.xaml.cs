using CameraViewLib;

namespace MauiAppDotNet7
{
    public partial class MainPage : ContentPage
    {
        int count = 0;

        public MainPage()
        {
            InitializeComponent();
            streaming.Clicked += OnStreamingButtonClicked;
        }

        private void OnCounterClicked(object sender, EventArgs e)
        {
            count++;

            if (count == 1)
                CounterBtn.Text = $"Clicked {count} time";
            else
                CounterBtn.Text = $"Clicked {count} times";

            SemanticScreenReader.Announce(CounterBtn.Text);
        }
        async void OnStreamingButtonClicked(object sender, EventArgs e)
        {
            streaming.IsEnabled = false;
            await Navigation.PushAsync(new CameraViewPage());
            streaming.IsEnabled = true;
        }
    }

}
