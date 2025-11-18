
template <int i>
struct compile_time_for{
    template <typename Lambda, typename... Args>
    inline static void op(const Lambda& function, Args&&... args) {
        compile_time_for<i-1>(function, std::forward<Args>(args)...);
        function(std::integral_constant<int, i - 1>{}, std::forward<Args>(args)...);
    }
};

template <>
struct compile_time_for<1>{
    template <typename Lambda, typename... Args>
    inline static void op(const Lambda& function, Args&&... args) {
        function(std::integral_constant<int, 0>{}, std::forward<Args>(args)...);
    }
};

