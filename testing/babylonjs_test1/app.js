window.addEventListener('DOMContentLoaded', function () {
    var canvas = document.getElementById('renderCanvas');
    var engine = new BABYLON.Engine(canvas, true);

    var createScene = function () {
        var scene = new BABYLON.Scene(engine);
        var camera = new BABYLON.ArcRotateCamera("Camera", Math.PI / 2, Math.PI / 2, 100, new BABYLON.Vector3(0, 0, 0), scene);        camera.attachControl(canvas, true);
        camera.inverted = true;

        var light = new BABYLON.HemisphericLight("light1", new BABYLON.Vector3(0, 1, 0), scene);

        scene.environmentTexture = BABYLON.CubeTexture.CreateFromPrefilteredData("babylon_assets/environmentSpecular.env", scene);

        BABYLON.SceneLoader.ImportMesh("", "models/building2/", "building3.babylon", scene, function (newMeshes) {
            camera.target = newMeshes[0];
        });

        return scene;
    };



    var scene = createScene();

    engine.runRenderLoop(function () {
        scene.render();
    });

    window.addEventListener('resize', function () {
        engine.resize();
    });

    scene.debugLayer.show();
});