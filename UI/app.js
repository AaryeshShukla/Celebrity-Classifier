Dropzone.autoDiscover = false;

function init() {
    let dz = new Dropzone("#dropzone", {
        maxFiles: 1,
        addRemoveLinks: true,
        dictDefaultMessage: "Drop image here or click to upload",
        autoProcessQueue: false,
        acceptedFiles: "image/*",
        clickable: true
    });

    dz.on("addedfile", function() {
        if (dz.files[1] != null) {
            dz.removeFile(dz.files[0]);        
        }
    });

    dz.on("complete", function(file) {
        let imageData = file.dataURL;
        var url = "http://127.0.0.1:5000/classify_image";

        $.post(url, { image_data: imageData }, function(data, status) {
            if (!data || data.length == 0) {
                $("#error").show();
                $("#divClassTable").hide();
                $("#resultHolder").hide();
                return;
            }

            let match = data[0]; // util.py returns list of 1 item
            $("#error").hide();
            $("#resultHolder").show();
            $("#divClassTable").show();
            $("#resultHolder").html($(`[data-player="${match.class}"]`).html());

            let probabilities = match.probabilities;
            for (let personName in probabilities) {
                let elementName = "#score_" + personName;
                $(elementName).html(probabilities[personName]);
            }
        });
    });

    $("#submitBtn").on('click', function () {
        dz.processQueue();		
    });
}

$(document).ready(function() {
    $("#error").hide();
    $("#resultHolder").hide();
    $("#divClassTable").hide();
    init();
});
